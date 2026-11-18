"""Training-only normalization, variable-batch grouped training and offline gates."""
from common import *
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from network import CandidateGainTransformer
from transformer_ppo.archive import ObjectiveNormalizer,exact_hv_3d

def normalizer(record):return ObjectiveNormalizer(tuple(record['minimum']),tuple(record['maximum']),tuple(record['reference']))
def episode_normalizer(record,div):return normalizer(record).with_divisor(div)
def fit_normalization(collections):
    values=[];sources=[]
    for folder in collections:
        result=read_json(folder/'DATA.json');div=np.asarray(result['divisor'])
        if result['role']!='training':raise ValueError('Normalization may use training instances only')
        values.extend(np.asarray(result['feasible_raw'],dtype=float).reshape(-1,3)/div)
        for row in result['rows']:
            for pair in row['labels']:
                values.extend(np.asarray([x['objectives'] for x in pair if x['feasible']],dtype=float).reshape(-1,3)/div)
        sources.append({'instance':result['instance'],'sha256':digest(folder/'DATA.json')})
    if not values:raise ValueError('No feasible training points')
    x=np.asarray(values);lo=x.min(0);hi=x.max(0);hi=np.maximum(hi,lo+1e-8)
    return {'source':'training-only','minimum':lo.tolist(),'maximum':hi.tolist(),'reference':[1.1]*3,
        'objective_divisor':'total quantity,total quantity,instance horizon','objective_clipping':False,
        'outside_reference':'points outside reference contribute no HV; objective coordinates are not clipped',
        'sources':sources,'feasible_point_count':len(x)}

def context(row,norm):
    n=episode_normalizer(norm,row['divisor']);points=np.asarray([n.transform(x) for x in row['population_raw']])
    summary=np.concatenate([points.mean(0),points.std(0),points.min(0),points.max(0)]) if len(points) else np.zeros(12)
    return np.concatenate([summary,[row['progress'],row['initialization_progress'],row['population_fraction']]])

def labels(row,norm):
    n=episode_normalizer(norm,row['divisor']);points=[n.transform(x) for x in row['population_raw']]
    h=exact_hv_3d(points,n.reference)
    q=np.asarray([exact_hv_3d(points+[n.transform(x['objectives']) for x in pair if x['feasible']],n.reference)-h for pair in row['labels']])
    return q-q[0]

def moments(x):
    x=np.asarray(x);s=x.std(0)
    return {'mean':x.mean(0).tolist(),'std':np.where(s>1e-8,s,1.).tolist()}
def scaler(rows,norm):
    return {'feature_version':FEATURE_VERSION,'batch':moments(np.concatenate([np.asarray(r['candidate_batch_tokens']).reshape(-1,len(r['candidate_batch_tokens'][0][0])) for r in rows])),
        'candidate':moments(np.concatenate([r['candidate_features'] for r in rows])),
        'context':moments([context(r,norm) for r in rows]),'clip':10.}
def pack(rows,s,norm):
    def scaled(x,part):return np.clip((np.asarray(x)-s[part]['mean'])/s[part]['std'],-s['clip'],s['clip']).astype(np.float32)
    batches=torch.from_numpy(np.stack([scaled(r['candidate_batch_tokens'],'batch') for r in rows]))
    candidates=torch.from_numpy(np.stack([scaled(r['candidate_features'],'candidate') for r in rows]))
    contexts=torch.from_numpy(scaled([context(r,norm) for r in rows],'context'))
    marker=torch.zeros(len(rows),8,1);marker[:,0,0]=1
    return batches,candidates,contexts,marker

def loss(pred,target,mask):
    delta=pred[:,:,None]-pred[:,None,:];truth=target[:,:,None]-target[:,None,:]
    valid=torch.triu(torch.ones((8,8),dtype=torch.bool),diagonal=1)[None]&mask[:,:,None]&mask[:,None,:]&(truth.abs()>1e-6)
    if not valid.any():return pred.sum()*0
    weights=truth[valid].abs().clamp(.1,5.)
    return (weights*nn.functional.softplus(-truth[valid].sign()*delta[valid])).sum()/weights.sum()

def predict(models,rows,s,norm,scale):
    result=np.zeros((len(models),len(rows),8))
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[len(r['candidate_batch_tokens'][0])].append(i)
    with torch.inference_mode():
        for indices in groups.values():
            for start in range(0,len(indices),64):
                ids=indices[start:start+64];t=pack([rows[i] for i in ids],s,norm)
                for j,model in enumerate(models):model.eval();result[j,ids]=model(*t).numpy()*scale
    return result

def decide(pred,mask,lam,threshold,disabled=False):
    if disabled:return 0
    scores=pred.mean(0)-lam*pred.std(0)
    scores=np.where(mask,scores,-np.inf);scores[0]=0
    i=int(np.argmax(scores));return i if i and scores[i]>threshold else 0

def validation_score(rows,preds,norm,lam,threshold,disabled=False,targets=None):
    group=defaultdict(lambda:defaultdict(list));switch=[]
    for i,row in enumerate(rows):
        a=decide(preds[:,i],row['legal_mask'],lam,threshold,disabled)
        target=labels(row,norm) if targets is None else targets[i]
        group[row['configuration']][row['instance']].append(float(target[a]));switch.append(a!=0)
    per={c:float(np.mean([np.mean(x) for x in instances.values()])) for c,instances in group.items()}
    return {'mean_gain':float(np.mean(list(per.values()))),'switch_fraction':float(np.mean(switch)),'per_configuration':per}

def train_component(component,train,valid,norm,settings,out):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True);s=scaler(train,norm)
    targets=np.stack([labels(r,norm) for r in train]);scale=max(float(np.std(targets)),1e-9)
    write_json(out/'INPUT_SCALER.json',s)
    groups=defaultdict(list)
    for i,r in enumerate(train):groups[len(r['candidate_batch_tokens'][0])].append(i)
    all_models={e:[] for e in settings['checkpoint_epochs']};history=[]
    dimensions=(len(s['batch']['mean']),len(s['candidate']['mean']),len(s['context']['mean']))
    for seed in settings['training_seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=CandidateGainTransformer(*dimensions,width=settings['width']);optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        packed={k:pack([train[i] for i in ids],s,norm) for k,ids in groups.items()}
        for epoch in range(1,settings['epochs']+1):
            model.train();losses=[]
            work=[]
            for k,ids in groups.items():
                order=rng.permutation(len(ids))
                for start in range(0,len(ids),settings['batch_size']):work.append((k,order[start:start+settings['batch_size']]))
            rng.shuffle(work)
            for k,local in work:
                ids=[groups[k][i] for i in local];t=tuple(x[local] for x in packed[k])
                y=torch.tensor(targets[ids]/scale,dtype=torch.float32);mask=torch.tensor([train[i]['legal_mask'] for i in ids],dtype=torch.bool)
                objective=loss(model(*t),y,mask);optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(objective))
            if epoch in settings['checkpoint_epochs']:
                path=out/f'seed_{seed}_epoch_{epoch}.pt';torch.save({'model_state':model.state_dict(),'dimensions':dimensions,'width':settings['width'],'feature_version':FEATURE_VERSION},path)
                clone=CandidateGainTransformer(*dimensions,width=settings['width']);clone.load_state_dict(model.state_dict());clone.eval();all_models[epoch].append(clone)
                history.append({'seed':seed,'epoch':epoch,'training_loss':float(np.mean(losses))})
                print(f'[B20-TRAIN] {component} seed={seed} epoch={epoch} loss={np.mean(losses):.6g}',flush=True)
    validation_targets=np.stack([labels(r,norm) for r in valid]);choices=[]
    for epoch,models in all_models.items():
        preds=predict(models,valid,s,norm,scale)
        for lam in settings['gate_lambda']:
            for threshold in settings['gate_threshold_scaled']:
                choices.append({'epoch':epoch,'lambda':lam,'threshold_scaled':threshold,'disabled':False,**validation_score(valid,preds,norm,lam,threshold*scale,targets=validation_targets)})
    choices.append({'epoch':settings['checkpoint_epochs'][0],'lambda':0.,'threshold_scaled':0.,'disabled':True,'mean_gain':0.,'switch_fraction':0.,'per_configuration':{c:0. for c in settings['configurations']}})
    best=min(choices,key=lambda x:(-x['mean_gain'],x['switch_fraction'],x['epoch'],x['lambda'],x['threshold_scaled']))
    paths=[out/f'seed_{seed}_epoch_{best["epoch"]}.pt' for seed in settings['training_seeds']]
    bundle={'protocol':settings['protocol'],'feature_version':FEATURE_VERSION,'component':component,'scale':scale,'gate':best,
        'normalization':norm,'scaler':s,'checkpoints':[x.name for x in paths],'sha256':{x.name:digest(x) for x in paths},
        'training_instances':sorted({r['instance'] for r in train}),'validation_instances':sorted({r['instance'] for r in valid}),
        'training_states':len(train),'validation_states':len(valid),'training_selection_seconds':time.perf_counter()-started}
    write_json(out/'VALIDATION.json',{'choices':choices,'selected':best,'history':history})
    write_json(out/'BUNDLE.json',bundle);return bundle

class FrozenPolicy:
    def __init__(self,folder):
        self.folder=Path(folder);self.bundle=read_json(self.folder/'BUNDLE.json');b=self.bundle
        if b['feature_version']!=FEATURE_VERSION:raise ValueError('Incompatible feature schema; old weights are forbidden')
        self.models=[];self.range_checks={k:{'values':0,'clipped_values':0,'max_abs_z':0.} for k in ('batch','candidate','context')}
        for name in b['checkpoints']:
            path=self.folder/name
            if digest(path)!=b['sha256'][name]:raise ValueError('Frozen checkpoint changed')
            import inspect
            extra={'weights_only':True} if 'weights_only' in inspect.signature(torch.load).parameters else {}
            ck=torch.load(path,map_location='cpu',**extra);m=CandidateGainTransformer(*ck['dimensions'],width=ck['width']);m.load_state_dict(ck['model_state']);m.eval();self.models.append(m)
    def select(self,row):
        b=self.bundle;s=b['scaler']
        for part,x in [('batch',row['candidate_batch_tokens']),('candidate',row['candidate_features']),('context',context(row,b['normalization']))]:
            z=np.abs((np.asarray(x)-s[part]['mean'])/s[part]['std']);r=self.range_checks[part]
            r['values']+=z.size;r['clipped_values']+=int(np.sum(z>s['clip']));r['max_abs_z']=max(r['max_abs_z'],float(z.max()))
        p=predict(self.models,[row],b['scaler'],b['normalization'],b['scale'])[:,0]
        g=b['gate'];return decide(p,row['legal_mask'],g['lambda'],g['threshold_scaled']*b['scale'],g['disabled'])
