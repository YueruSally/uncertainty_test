"""Masked shared actor-critic, with frozen supervision gates and terminal returns."""
from bridge import *
from collections import defaultdict
import copy
import time


class ActorCritic(SharedTransformer):
    def __init__(self,dimensions,width=16):
        super().__init__(dimensions,width);self.value_head=nn.Linear(width,1)
        nn.init.zeros_(self.value_head.weight);nn.init.zeros_(self.value_head.bias)
    def forward(self,c,batch,candidate,context,marker):
        b,k,n,f=batch.shape
        h=self.batch_encoder(self.batch_projection[c](batch.reshape(b*k,n,f))).mean(1).reshape(b,k,-1)
        h=self.candidate_projection[c](torch.cat((h,candidate,context[:,None,:].expand(-1,k,-1),marker),-1))
        h=self.candidate_encoder(h+self.task.weight[COMPONENTS.index(c)])
        raw=self.gain_head(h).squeeze(-1)
        return torch.cat((torch.zeros_like(raw[:,:1]),raw[:,1:]),1),self.value_head(h.mean(1)).squeeze(-1)


class TerminalPolicy:
    def __init__(self,folder,sample=False,seed=0,record=False):
        self.bundle=read_json(folder/'BUNDLE.json');b=self.bundle
        if b['feature_version']!=FEATURE:raise ValueError('Incompatible PPO inputs')
        verify({str(folder/p):sha for p,sha in b['checkpoint_sha256'].items()})
        self.models=[]
        for p in b['checkpoints']:
            m=ActorCritic(b['dimensions'],b['settings']['width'])
            missing,unexpected=m.load_state_dict(load_checkpoint(folder/p)['model'],strict=False)
            if unexpected or set(missing)-{'value_head.weight','value_head.bias'}:raise ValueError('Incompatible actor checkpoint')
            if b.get('terminal_checkpoint') and missing:raise ValueError('Incomplete critic checkpoint')
            m.eval();self.models.append(m)
        self.sample=sample;self.record=record;self.rng=np.random.default_rng(seed)
        self.events=defaultdict(list);self.range_checks={c:{'values':0,'clipped':0} for c in COMPONENTS}

    def distribution(self,c,x,mask):
        outputs=[m(c,*x) for m in self.models];pred=torch.stack([v[0] for v in outputs])
        gate=self.bundle['selected']['gates'][c]
        score=pred.mean(0)-gate['lambda']*pred.std(0,unbiased=False)
        score=torch.cat((torch.zeros_like(score[:,:1]),score[:,1:]-gate['threshold_scaled']),1)
        mask=mask.clone()
        if gate['disabled']:mask[:,1:]=False
        score=score.masked_fill(~mask,-torch.inf)
        probs=.99*torch.softmax(score,-1)+.01*mask.float()/mask.sum(-1,keepdim=True)
        return torch.distributions.Categorical(probs=probs),torch.stack([v[1] for v in outputs]).mean(0)

    def select(self,row):
        c=row['component'];b=self.bundle;s=b['scalers'][c]
        row=progress_row(row,b['settings']['progress_denominator'])
        for part,x in [('batch',row['candidate_batch_tokens']),('candidate',row['candidate_features']),('context',legacy_learning.context(row,b['normalization']))]:
            z=np.abs((np.asarray(x)-s[part]['mean'])/s[part]['std'])
            self.range_checks[c]['values']+=int(z.size);self.range_checks[c]['clipped']+=int((z>s['clip']).sum())
        x=legacy_learning.pack([row],s,b['normalization'])
        mask=torch.tensor([row['legal_mask']],dtype=torch.bool)
        with torch.no_grad():
            dist,value=self.distribution(c,x,mask)
            prob=dist.probs[0].numpy().astype(float);prob/=prob.sum()
            action=int(self.rng.choice(8,p=prob)) if self.sample else int(prob.argmax())
            lp=float(dist.log_prob(torch.tensor([action])).item())
        if not row['legal_mask'][action]:raise ValueError('Invalid chosen candidate')
        if self.record:
            self.events[c].append({'x':tuple(t[0].numpy().copy() for t in x),'mask':mask[0].numpy().copy(),
                                   'action':action,'logp':lp,'value':float(value.item())})
        return action

    def save_events(self,path):
        arrays={}
        for c,events in self.events.items():
            for j,k in enumerate(('batch','candidate','context','marker')):
                arrays[c+'_'+k]=np.stack([r['x'][j] for r in events])
            for k in ('mask','action','logp','value'):arrays[c+'_'+k]=np.asarray([r[k] for r in events])
        np.savez_compressed(path,**arrays)


def optimizer_for(policy,settings):
    return torch.optim.AdamW([p for m in policy.models for p in m.parameters()],lr=settings['learning_rate'],weight_decay=0.)


def update(policy,episodes,settings,optimizer,seed):
    """Each episode has equal total weight; terminal reward, gamma=1."""
    loaded=[];indices=defaultdict(list)
    for e,(file,reward) in enumerate(episodes):
        with np.load(file,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        count=sum(len(arrays.get(c+'_action',[])) for c in COMPONENTS)
        if not count:raise ValueError('No actions in PPO episode')
        loaded.append((arrays,float(reward),1./len(episodes)/count))
        for c in COMPONENTS:
            if c+'_action' not in arrays:continue
            for i in range(len(arrays[c+'_action'])):indices[c,arrays[c+'_batch'].shape[2]].append((e,i))
    total=sum(map(len,indices.values()));max_error=0.;steps=0;last_kl=0.;stopped=False
    def batch(c,ids):
        x=tuple(torch.tensor(np.stack([loaded[e][0][c+'_'+k][i] for e,i in ids]),dtype=torch.float32)
                for k in ('batch','candidate','context','marker'))
        def values(k,dtype):return torch.tensor([loaded[e][0][c+'_'+k][i] for e,i in ids],dtype=dtype)
        mask=torch.tensor(np.stack([loaded[e][0][c+'_mask'][i] for e,i in ids]),dtype=torch.bool)
        return x,mask,values('action',torch.long),values('logp',torch.float32),values('value',torch.float32),torch.tensor([loaded[e][1] for e,i in ids]),torch.tensor([loaded[e][2]*total for e,i in ids])
    with torch.no_grad():
        for (c,n),ids in indices.items():
            for start in range(0,len(ids),settings['minibatch']):
                x,mask,a,old,*_=batch(c,ids[start:start+settings['minibatch']])
                dist,_=policy.distribution(c,x,mask)
                max_error=max(max_error,float((dist.log_prob(a)-old).abs().max()))
    if max_error>3e-5:raise ValueError(f'Off-policy probabilities: {max_error}')
    rng=np.random.default_rng(seed)
    for epoch in range(settings['epochs_per_update']):
        work=[]
        for (c,n),ids in indices.items():
            shuffled=[ids[int(i)] for i in rng.permutation(len(ids))]
            work.extend((c,shuffled[i:i+settings['minibatch']]) for i in range(0,len(ids),settings['minibatch']))
        rng.shuffle(work)
        for c,ids in work:
            x,mask,a,old,old_value,ret,weight=batch(c,ids)
            dist,value=policy.distribution(c,x,mask);lp=dist.log_prob(a);ratio=(lp-old).exp()
            last_kl=float(((ratio-1)-(lp-old)).mean().detach())
            if last_kl>settings['target_kl']:stopped=True;break
            advantage=ret-old_value
            actor=-torch.minimum(ratio*advantage,ratio.clamp(1-settings['clip_ratio'],1+settings['clip_ratio'])*advantage)
            loss=(weight*(actor+settings['value_coefficient']*(value-ret)**2-settings['entropy_coefficient']*dist.entropy())).mean()
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_([p for m in policy.models for p in m.parameters()],settings['max_grad_norm'])
            optimizer.step();steps+=1
            if steps%100==0:print(f'[MULTI-PPO] optimizer_steps={steps} kl={last_kl:.6g}',flush=True)
        if stopped:break
    return {'optimizer_steps':steps,'kl_early_stop':stopped,'last_minibatch_kl':last_kl,
            'on_policy_logp_max_error':max_error,'actions':total,'complete_episodes':len(episodes),
            'terminal_returns':[float(r) for _,r in episodes]}


def snapshot(policy,folder,optimizer=None,metrics=None):
    folder.parent.mkdir(parents=True,exist_ok=True)
    stage=folder.with_name(folder.name+f'_building_{time.time_ns()}');stage.mkdir()
    b=copy.deepcopy(policy.bundle);names=[]
    for i,m in enumerate(policy.models):
        p=stage/f'member_{i}.pt';save_checkpoint(p,{'model':m.state_dict()});names.append(p.name)
    b.update(checkpoints=names,checkpoint_sha256={p:digest(stage/p) for p in names},terminal_checkpoint=True)
    if optimizer:
        save_checkpoint(stage/'optimizer.pt',optimizer.state_dict());b['optimizer_sha256']=digest(stage/'optimizer.pt')
    write_json(stage/'BUNDLE.json',b)
    if metrics is not None:write_json(stage/'UPDATE.json',metrics)
    stage.rename(folder)
