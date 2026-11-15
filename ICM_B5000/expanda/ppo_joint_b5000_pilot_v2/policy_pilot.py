"""Paired-terminal PPO with balanced tasks and explicit deployment controls."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
TERMINAL = HERE.parent / 'joint_icm_multiscale_terminal_v1'
sys.path.insert(0, str(TERMINAL))
from bridge import *
from policy import TerminalPolicy, snapshot, ActorCritic
from collections import defaultdict
HERE = Path(__file__).resolve().parent


class DetachedCritic(ActorCritic):
    def forward(self, c, batch, candidate, context, marker):
        b,k,n,f = batch.shape
        h = self.batch_encoder(self.batch_projection[c](batch.reshape(b*k,n,f))).mean(1).reshape(b,k,-1)
        h = self.candidate_projection[c](torch.cat((h,candidate,context[:,None,:].expand(-1,k,-1),marker),-1))
        h = self.candidate_encoder(h+self.task.weight[COMPONENTS.index(c)])
        raw = self.gain_head(h).squeeze(-1)
        scores = torch.cat((torch.zeros_like(raw[:,:1]),raw[:,1:]),1)
        return scores, self.value_head(h.detach().mean(1)).squeeze(-1)


class PilotPolicy(TerminalPolicy):
    def __init__(self, folder, sample=False, seed=0, record=False):
        super().__init__(folder, sample, seed, record)
        models=[]
        for old in self.models:
            m=DetachedCritic(self.bundle['dimensions'],self.bundle['settings']['width'])
            m.load_state_dict(old.state_dict());m.eval();models.append(m)
        self.models=models

    def scores(self, c, x, mask, return_predictions=False):
        outputs=[m(c,*x) for m in self.models]
        pred=torch.stack([z[0] for z in outputs]);g=self.bundle['selected']['gates'][c]
        score=pred.mean(0)-g['lambda']*pred.std(0,unbiased=False)
        score=torch.cat((torch.zeros_like(score[:,:1]),score[:,1:]-g['threshold_scaled']),1)
        legal=mask.clone()
        if g['disabled']:legal[:,1:]=False
        result=(score.masked_fill(~legal,-torch.inf),legal,torch.stack([z[1] for z in outputs]).mean(0))
        return result+(pred,) if return_predictions else result

    def distribution_from_scores(self, c, score, legal):
        spec=self.bundle.get('pilot',{})
        temperature=spec.get('temperatures',{}).get(c,1.)
        epsilon=spec.get('uniform_exploration',.01)
        prob=(1-epsilon)*torch.softmax(score/temperature,-1)+epsilon*legal.float()/legal.sum(-1,keepdim=True)
        return torch.distributions.Categorical(probs=prob)

    def distribution(self, c, x, mask):
        score,legal,value=self.scores(c,x,mask)
        return self.distribution_from_scores(c,score,legal),value

    def select(self,row):
        c=row['component'];b=self.bundle;s=b['scalers'][c]
        row=progress_row(row,b['settings']['progress_denominator'])
        for part,v in [('batch',row['candidate_batch_tokens']),('candidate',row['candidate_features']),('context',legacy_learning.context(row,b['normalization']))]:
            z=np.abs((np.asarray(v)-s[part]['mean'])/s[part]['std'])
            self.range_checks[c]['values']+=int(z.size);self.range_checks[c]['clipped']+=int((z>s['clip']).sum())
        x=legacy_learning.pack([row],s,b['normalization'])
        mask=torch.tensor([row['legal_mask']],dtype=torch.bool)
        with torch.no_grad():
            score,legal,value,pred=self.scores(c,x,mask,return_predictions=True)
            dist=self.distribution_from_scores(c,score,legal)
            p=dist.probs[0].numpy().astype(float);p/=p.sum()
            # Greedy uses the original shared selector, not temperature-rounded probabilities.
            if self.sample:action=int(self.rng.choice(8,p=p))
            else:
                pred=pred[:,0].numpy().astype(np.float64);g=b['selected']['gates'][c]
                action=int(legacy_learning.decide(pred,row['legal_mask'],g['lambda'],g['threshold_scaled'],g['disabled']))
            logp=float(dist.log_prob(torch.tensor([action])).item())
        if not row['legal_mask'][action]:raise ValueError('Illegal action')
        if self.record:self.events[c].append(dict(x=tuple(t[0].numpy().copy() for t in x),mask=mask[0].numpy().copy(),
            action=action,logp=logp,value=float(value.item()),prob=dist.probs[0].numpy().copy()))
        return action

    def save_events(self,path):
        arrays={}
        for c,rows in self.events.items():
            for j,k in enumerate(('batch','candidate','context','marker')):arrays[c+'_'+k]=np.stack([r['x'][j] for r in rows])
            for k in ('mask','action','logp','value','prob'):arrays[c+'_'+k]=np.asarray([r[k] for r in rows])
        np.savez_compressed(path,**arrays)


def pack_arrays(arrays,c,ids):
    x=tuple(torch.tensor(arrays[c+'_'+k][ids],dtype=torch.float32) for k in ('batch','candidate','context','marker'))
    mask=torch.tensor(arrays[c+'_mask'][ids],dtype=torch.bool)
    return x,mask


def calibrate(source_model, paths, spec):
    """Only pre-existing TRAINING candidate states; no objective values or validation labels."""
    policy=PilotPolicy(source_model);samples={c:[] for c in COMPONENTS};provenance={}
    for path in paths:
        provenance[str(path)]=digest(path)
        with np.load(path,allow_pickle=False) as saved:
            for c in COMPONENTS:
                if c+'_action' not in saved:raise ValueError('Calibration trajectory lacks '+c)
                count=len(saved[c+'_action']);n=min(count,spec['calibration_states_per_component_per_configuration'])
                ids=np.linspace(0,count-1,n,dtype=int)
                arrays={c+'_'+k:saved[c+'_'+k][ids] for k in ('batch','candidate','context','marker','mask')}
                x,mask=pack_arrays(arrays,c,np.arange(n))
                with torch.no_grad():scores,legal,_=policy.scores(c,x,mask)
                samples[c].append((scores,legal))
    temperatures={};report={}
    for c,groups in samples.items():
        options=[]
        for temperature in spec['temperature_grid']:
            tops=[];entropies=[]
            for scores,mask in groups:
                epsilon=spec['uniform_exploration']
                prob=(1-epsilon)*torch.softmax(scores/temperature,-1)+epsilon*mask.float()/mask.sum(-1,keepdim=True)
                tops.extend(prob.max(-1).values.tolist());entropies.extend(torch.distributions.Categorical(probs=prob).entropy().tolist())
            options.append(dict(temperature=temperature,mean_top_probability=float(np.mean(tops)),mean_entropy=float(np.mean(entropies))))
        chosen=min(options,key=lambda r:(abs(r['mean_top_probability']-spec['calibration_target_mean_top_probability']),-r['temperature']))
        temperatures[c]=chosen['temperature'];report[c]=dict(selected=chosen,states=sum(len(x) for x,_ in groups),candidates=options)
    return temperatures,dict(components=report,files=provenance,source='training candidate states only; unchanged greedy argmax')


def advantage_statistics(arrays, returns):
    stats={}
    for c in COMPONENTS:
        values=[float(r)-a[c+'_value'].astype(np.float64) for a,r in zip(arrays,returns)]
        if any(len(v)==0 for v in values):raise ValueError('Each training episode must contain all three components')
        mean=float(np.mean([v.mean() for v in values]));second=float(np.mean([(v*v).mean() for v in values]))
        std=float(np.sqrt(max(0.,second-mean*mean)));denominator=max(std,1e-8)
        z=[(v-mean)/denominator for v in values]
        stats[c]=dict(mean=mean,std=std,denominator=denominator,normalized_mean=float(np.mean([v.mean() for v in z])),
            normalized_second_moment=float(np.mean([(v*v).mean() for v in z])),actions=sum(len(v) for v in values))
    return stats


def optimizer_for(policy,spec):
    actor=[];critic=[]
    for model in policy.models:
        for name,p in model.named_parameters():(critic if name.startswith('value_head.') else actor).append(p)
    return torch.optim.AdamW([{'params':actor,'lr':spec['learning_rate']},{'params':critic,'lr':spec['critic_learning_rate']}],weight_decay=0.)


def categorical_kl(p,q):
    return (p*(p.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(-1)


def update(policy,reference,episodes,spec,optimizer,seed):
    arrays=[];returns=[]
    for file,reward in episodes:
        with np.load(file,allow_pickle=False) as saved:arrays.append({k:saved[k] for k in saved.files})
        returns.append(float(reward))
    stats=advantage_statistics(arrays,returns);rng=np.random.default_rng(seed);max_error=0.
    # Verify every saved action is on-policy; cache the fixed reference distribution.
    with torch.no_grad():
        for a in arrays:
            for c in COMPONENTS:
                cached=[]
                for start in range(0,len(a[c+'_action']),64):
                    ids=np.arange(start,min(start+64,len(a[c+'_action'])));x,mask=pack_arrays(a,c,ids)
                    dist,_=policy.distribution(c,x,mask);ref,_=reference.distribution(c,x,mask)
                    old=torch.tensor(a[c+'_logp'][ids],dtype=torch.float32);action=torch.tensor(a[c+'_action'][ids],dtype=torch.long)
                    max_error=max(max_error,float((dist.log_prob(action)-old).abs().max()))
                    if not np.allclose(dist.probs.numpy(),a[c+'_prob'][ids],rtol=1e-5,atol=1e-7):raise ValueError('Stored probabilities changed')
                    cached.append(ref.probs.numpy())
                a[c+'_reference_prob']=np.concatenate(cached)
    if max_error>3e-5:raise ValueError('Off-policy actions: '+str(max_error))
    before=[{k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('value_head.')} for m in policy.models]
    orders={c:rng.permutation(len(arrays)).tolist() for c in COMPONENTS};completed=0;stopped=False;history=[]
    for step in range(spec['optimizer_steps']):
        optimizer.zero_grad();losses=[];metrics={}
        for c in COMPONENTS:
            e=orders[c][step%len(arrays)];a=arrays[e];ids=rng.integers(0,len(a[c+'_action']),size=spec['minibatch_per_component'])
            x,mask=pack_arrays(a,c,ids);dist,value=policy.distribution(c,x,mask)
            actions=torch.tensor(a[c+'_action'][ids],dtype=torch.long);old=torch.tensor(a[c+'_logp'][ids],dtype=torch.float32)
            oldprob=torch.tensor(a[c+'_prob'][ids],dtype=torch.float32);refprob=torch.tensor(a[c+'_reference_prob'][ids],dtype=torch.float32)
            advantage=(returns[e]-torch.tensor(a[c+'_value'][ids],dtype=torch.float32)-stats[c]['mean'])/stats[c]['denominator']
            ratio=(dist.log_prob(actions)-old).exp();kl=categorical_kl(oldprob,dist.probs).mean()
            actor=-torch.minimum(ratio*advantage,ratio.clamp(1-spec['clip_ratio'],1+spec['clip_ratio'])*advantage).mean()
            anchor=categorical_kl(refprob,dist.probs).mean();critic=(value-returns[e]).square().mean()
            losses.append(actor+spec['reference_kl_coefficient']*anchor+spec['value_coefficient']*critic)
            metrics[c]=dict(kl=float(kl.detach()),reference_kl=float(anchor.detach()),entropy=float(dist.entropy().mean().detach()),
                clip_fraction=float(((ratio-1).abs()>spec['clip_ratio']).float().mean()),episode=e)
        if max(v['kl'] for v in metrics.values())>spec['target_kl']:
            stopped=True;history.append(dict(step=step,applied=False,components=metrics));break
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss')
        loss.backward();grad=nn.utils.clip_grad_norm_([p for m in policy.models for p in m.parameters()],spec['max_grad_norm'])
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
        optimizer.step();completed+=1;history.append(dict(step=step,applied=True,components=metrics))
    change=sum(float((m.state_dict()[k]-v).square().sum()) for m,b in zip(policy.models,before) for k,v in b.items())**.5
    diagnostic={}
    with torch.no_grad():
        for c in COMPONENTS:
            switches=[];prob_changes=[];ref_margins=[];kl_values=[]
            for a in arrays:
                count=len(a[c+'_action']);ids=np.linspace(0,count-1,min(count,64),dtype=int);x,mask=pack_arrays(a,c,ids)
                dist,_=policy.distribution(c,x,mask);ref=torch.tensor(a[c+'_reference_prob'][ids])
                choice=ref.argmax(-1);idx=torch.arange(len(ids));top=ref.topk(min(2,ref.shape[-1]),dim=-1).values
                switches.extend((dist.probs.argmax(-1)!=choice).float().tolist())
                prob_changes.extend((dist.probs[idx,choice]-ref[idx,choice]).tolist());ref_margins.extend((top[:,0]-top[:,1]).tolist())
                kl_values.extend(categorical_kl(ref,dist.probs).tolist())
            diagnostic[c]=dict(same_pool_switch_fraction=float(np.mean(switches)),u0_choice_probability_change=float(np.mean(prob_changes)),
                u0_top_two_margin=float(np.mean(ref_margins)),kl_u0_to_new=float(np.mean(kl_values)))
    return dict(optimizer_steps=completed,kl_early_stop=stopped,on_policy_logp_max_error=max_error,
        paired_terminal_returns=returns,advantage=stats,component_minibatches={c:completed for c in COMPONENTS},
        actor_parameter_l2_change=change,update_history=history,same_training_pool_diagnostics=diagnostic,
        diagnostic_scope='training states only, not validation performance')
