"""Configuration/task-balanced shared supervision with train-only scaling."""
from bootstrap import *
from shared import *
import csv
import math
import time


def balanced_groups(rows, configurations):
    groups={}
    for c in COMPONENTS:
        for cfg in configurations:
            indices=[i for i,r in enumerate(rows[c]['training']) if r['configuration']==cfg]
            if not indices: raise ValueError('Empty training task/configuration: '+c+'/'+cfg)
            if len({len(rows[c]['training'][i]['candidate_batch_tokens'][0]) for i in indices})!=1:
                raise ValueError('Mixed batch lengths within one configuration')
            groups[c,cfg]=indices
    return groups


def choose_gates(rows, pred, targets, scale, settings):
    choices=[]
    for lam in settings['gate_lambda']:
        for threshold in settings['gate_threshold_scaled']:
            q=legacy_learning.validation_score(rows,pred,None,lam,threshold,targets=targets/scale)
            choices.append(dict(q,lambda_value=lam,threshold_scaled=threshold,disabled=False))
    choices.append({'mean_gain':0.,'switch_fraction':0.,'per_configuration':{c:0. for c in settings['configurations']},
                    'lambda_value':0.,'threshold_scaled':0.,'disabled':True})
    best=min(choices,key=lambda v:(-v['mean_gain'],v['switch_fraction'],v['lambda_value'],v['threshold_scaled'],not v['disabled']))
    best['lambda']=best.pop('lambda_value')
    best['mean_raw_gain']=best['mean_gain']*scale
    best['per_configuration_raw_gain']={c:x*scale for c,x in best['per_configuration'].items()}
    return best


def fit(rows,norm,settings,out,provenance,coverage):
    out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); torch.set_num_threads(1)
    train_names={c:sorted({r['instance'] for r in rows[c]['training']}) for c in COMPONENTS}
    valid_names={c:sorted({r['instance'] for r in rows[c]['validation']}) for c in COMPONENTS}
    for c in COMPONENTS:
        if set(train_names[c])&set(valid_names[c]):raise ValueError('Training/validation overlap')
        if train_names[c]!=train_names['I'] or valid_names[c]!=valid_names['I']:raise ValueError('Different component splits')
    metadata={'settings':settings,'files':provenance,'coverage':coverage,'feature_version':FEATURE,
              'training_instances':train_names,'validation_instances':valid_names,
              'environment':{'torch':str(torch.__version__),'numpy':str(np.__version__)}}
    freeze(out/'PLAN.json',metadata)
    if (out/'BUNDLE.json').exists():
        b=read_json(out/'BUNDLE.json');SharedPolicy(out)
        if b['plan_sha256']!=digest(out/'PLAN.json'):raise ValueError('Model plan changed')
        return b
    freeze(out/'HV_NORMALIZATION.json',norm)
    # HV normalization -> normalized population context -> train-only feature
    # moments -> standardized inputs. Raw token/candidate features have no labels.
    scalers={c:legacy_learning.scaler(rows[c]['training'],norm) for c in COMPONENTS}
    for s in scalers.values():s['feature_version']=FEATURE
    freeze(out/'INPUT_SCALER.json',scalers)
    dims={c:[len(scalers[c][p]['mean']) for p in ('batch','candidate','context')] for c in COMPONENTS}
    targets={}
    for c in COMPONENTS:
        targets[c]={}
        for role in ('training','validation'):
            values=[]
            for i,r in enumerate(rows[c][role]):
                values.append(legacy_learning.labels(r,norm))
                if i%250==0:print(f'[MULTI-LABELS] {c} {role} {i}/{len(rows[c][role])}',flush=True)
            targets[c][role]=np.stack(values)
    scales={c:max(float(np.std(targets[c]['training'])),1e-9) for c in COMPONENTS}
    freeze(out/'LABEL_SCALE.json',scales)
    groups=balanced_groups(rows,settings['configurations'])
    packed={(c,cfg):legacy_learning.pack([rows[c]['training'][i] for i in ids],scalers[c],norm)
            for (c,cfg),ids in groups.items()}
    steps=max(math.ceil(len(ids)/settings['batch_size']) for ids in groups.values())
    for seed in settings['training_seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=SharedTransformer(dims,settings['width'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
        resume=out/f'resume_{seed}.pt';first=1
        if resume.exists():
            ck=load_checkpoint(resume)
            if ck['plan_sha256']!=digest(out/'PLAN.json'):raise ValueError('Resume belongs to another training plan')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            rng.bit_generator.state=ck['rng'];first=ck['epoch']+1
        for epoch in range(first,settings['epochs']+1):
            model.train();losses=[]
            schedules={key:[rng.choice(len(ids),size=min(settings['batch_size'],len(ids)),replace=False)
                            for _ in range(steps)] for key,ids in groups.items()}
            for step in range(steps):
                optimizer.zero_grad();total=0.
                # Every gradient step weights each configuration and component
                # equally, even though C has half as many sampled states.
                for (c,cfg),ids in groups.items():
                    local=schedules[c,cfg][step];selected=[ids[int(i)] for i in local]
                    x=tuple(t[local] for t in packed[c,cfg])
                    y=torch.tensor(targets[c]['training'][selected]/scales[c],dtype=torch.float32)
                    mask=torch.tensor([rows[c]['training'][i]['legal_mask'] for i in selected],dtype=torch.bool)
                    loss=legacy_learning.loss(model(c,*x),y,mask)/len(groups)
                    loss.backward();total+=float(loss.detach())
                nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(total)
            value={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                   'rng':rng.bit_generator.state,'plan_sha256':digest(out/'PLAN.json')}
            if epoch in settings['checkpoint_epochs']:save_checkpoint(out/f'seed_{seed}_epoch_{epoch}.pt',value)
            save_checkpoint(resume,value)
            if epoch%5==0 or epoch in settings['checkpoint_epochs']:
                print(f'[MULTI-SUP] seed={seed} epoch={epoch}/{settings["epochs"]} loss={np.mean(losses):.6g}',flush=True)
    candidates=[]
    for epoch in settings['checkpoint_epochs']:
        models=[]
        for seed in settings['training_seeds']:
            model=SharedTransformer(dims,settings['width'])
            model.load_state_dict(load_checkpoint(out/f'seed_{seed}_epoch_{epoch}.pt')['model']);models.append(model)
        gates={}
        for c in COMPONENTS:
            p=predictions(models,c,rows[c]['validation'],scalers[c],norm)
            gates[c]=choose_gates(rows[c]['validation'],p,targets[c]['validation'],scales[c],settings)
        candidates.append({'epoch':epoch,'score':float(np.mean([g['mean_gain'] for g in gates.values()])),
                           'switch_fraction':float(np.mean([g['switch_fraction'] for g in gates.values()])), 'gates':gates})
        print(f'[MULTI-VALID] epoch={epoch} score={candidates[-1]["score"]:.6g}',flush=True)
    selected=min(candidates,key=lambda x:(-x['score'],x['switch_fraction'],x['epoch']))
    paths=[f'seed_{seed}_epoch_{selected["epoch"]}.pt' for seed in settings['training_seeds']]
    bundle={**metadata,'plan_sha256':digest(out/'PLAN.json'),'normalization':norm,'scalers':scalers,
            'dimensions':dims,'target_scales':scales,'selected':selected,'checkpoints':paths,
            'checkpoint_sha256':{p:digest(out/p) for p in paths},
            'training_states':{c:len(rows[c]['training']) for c in COMPONENTS},
            'validation_states':{c:len(rows[c]['validation']) for c in COMPONENTS},
            'loss':'legal-pair magnitude-weighted softplus ranking; NO regression; task/configuration equal',
            'new_ppo_updates':0,'training_seconds_this_invocation':time.monotonic()-started}
    write_json(out/'VALIDATION.json',{'candidates':candidates,'selected':selected})
    write_json(out/'BUNDLE.json',bundle)
    report={'completed':True,'stage':'shared_supervised','coverage':coverage,'selected':selected,
            'training_states':bundle['training_states'],'validation_states':bundle['validation_states'],
            'new_training_objective_evaluations':0,'new_ppo_updates':0,'final_test_opened':False,
            'qualification':'Validation used for selection; raw gains are normalized single-step HV differences, NOT final HV percentages.'}
    write_json(out/'QUICK_RESULT.json',report)
    with (out/'00_共享监督验证.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['组件','配置','训练状态','验证状态','验证平均HV净增量_非百分比','切换比例_全部验证','门控关闭','选中轮次'])
        for c in COMPONENTS:
            g=selected['gates'][c]
            for cfg in settings['configurations']:
                w.writerow([c,cfg,sum(r['configuration']==cfg for r in rows[c]['training']),
                            sum(r['configuration']==cfg for r in rows[c]['validation']),
                            g['per_configuration_raw_gain'][cfg],g['switch_fraction'],g['disabled'],selected['epoch']])
    print('[MULTI-SUP] COMPLETE '+str(out),flush=True)
    return bundle
