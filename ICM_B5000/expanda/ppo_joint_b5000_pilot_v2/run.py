#!/usr/bin/env python3
"""A bounded development pilot. Original models, plans and results remain read-only."""
from policy_pilot import *
import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import signal
import socket
import resource
import time

B5=HERE.parent/'component_joint_b5000_v1'
sys.path.insert(0,str(B5))
from search_b5000 import BudgetController
from transformer_ppo.episode import run_episode,_optimisation_vector
from transformer_ppo.archive import nondominated
import baseline_uncertainty as base
spec=importlib.util.spec_from_file_location('pilot_scheduler',TERMINAL/'run.py')
scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler);scheduler.HERE2=HERE
spec=importlib.util.spec_from_file_location('pilot_hv',HERE.parent/'hv_joint_b5000_union_v1/recompute.py')
hv=importlib.util.module_from_spec(spec);spec.loader.exec_module(hv)


def check_pair(row,reference):
    for key in ('instance','seed','configuration','instance_digest','scenario_digest','evaluation_budget'):
        if row[key]!=reference[key]:raise ValueError('Unpaired '+key)
    if not row['completed'] or not reference['completed']:raise ValueError('Incomplete pair')


def prepare(root,source):
    old=read_json(source/'PLAN.json');protocol=read_json(HERE/'protocol.json')
    if canonical_digest({k:v for k,v in old.items() if k!='sha256'})!=old['sha256']:raise ValueError('Source plan changed')
    verify(old['files']);s=copy.deepcopy(old['settings'])
    if (s['budget'],s['population'],s['scenario_count'],s['progress_denominator'])!=(5000,250,100,50000):
        raise ValueError('Expected completed pop250/B5000 source with unchanged /50000 features')
    if not read_json(source/'STATUS.json').get('completed'):raise ValueError('Source experiment not complete')
    shared=Path(old['source'])/'shared_model';shared_bundle=read_json(shared/'BUNDLE.json');SharedPolicy(shared)
    if shared_bundle['normalization']!=old['normalization']:raise ValueError('Normalization mismatch')
    cached_model=source/'terminal/checkpoints/update_000'
    cached_policy=TerminalPolicy(cached_model)
    initial_policy=PilotPolicy(shared)
    for current,cached in zip(initial_policy.models,cached_policy.models):
        if any(not torch.equal(v,cached.state_dict()[k]) for k,v in current.state_dict().items()):
            raise ValueError('Cached u0 is not the unchanged shared initialization')
    if len(initial_policy.models)!=len(cached_policy.models):raise ValueError('Cached u0 ensemble differs')
    for key in ('normalization','scalers','selected','dimensions'):
        if initial_policy.bundle[key]!=cached_policy.bundle[key]:raise ValueError('Cached u0 input/gate mismatch: '+key)
    train=[];tune=[];confirmation=[]
    for c in protocol['configurations']:
        training=sorted((r for r in old['entries'] if r['configuration']==c and r['development_role']=='training'),key=lambda r:r['instance_id'])
        validation=sorted((r for r in old['entries'] if r['configuration']==c and r['development_role']=='validation'),key=lambda r:r['instance_id'])
        if len(training)!=6 or len(validation)!=2:raise ValueError('Unexpected source split')
        train.extend(training[:protocol['updates']]);tune.append(validation[0]);confirmation.append(validation[1])
    entries=train+tune+confirmation
    if len({r['instance_id'] for r in entries})!=len(entries) or len({r['batch_signature'] for r in entries})!=len(entries):raise ValueError('Overlapping instances')
    filemap={**old['files'],str(source/'PLAN.json'):digest(source/'PLAN.json')}
    filemap[str(cached_model/'BUNDLE.json')]=digest(cached_model/'BUNDLE.json')
    filemap.update({str(cached_model/p):sha for p,sha in cached_policy.bundle['checkpoint_sha256'].items()})
    for p in list(HERE.glob('*.py'))+[HERE/'protocol.json',HERE.parent/'hv_joint_b5000_union_v1/recompute.py']:
        filemap[str(p.resolve())]=digest(p)
    for entry in entries:
        original=source/'instances'/(entry['instance_id']+'.json');info,_=read_instance(original)
        if info!=entry:raise ValueError('Instance changed')
        freeze(root/'instances'/original.name,entry);filemap[str(original)]=digest(original)
        filemap[str(root/'instances'/original.name)]=digest(root/'instances'/original.name)
    seeds=s['ppo']['validation_seeds']
    if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Expected three frozen validation seeds')
    baseline_files=[]
    for entry in tune+confirmation:
        for seed in seeds:
            p=source/'terminal/validation/u000'/entry['instance_id']/f'seed_{seed}'/'RUN_COMPLETE.json'
            r=read_json(p)
            if (not r['completed'] or r['phase']!='validation' or r['stochastic'] or r['plan_sha256']!=old['sha256']
                or r['instance_digest']!=entry['instance_digest'] or r['seed']!=seed or r['evaluation_budget']!=5000
                or r['model_bundle_sha256']!=filemap[str(cached_model/'BUNDLE.json')]
                or r['hidden_candidate_objective_evaluations'] or r['real_objective_evaluation_count'] not in (4999,5000)):
                raise ValueError('Invalid cached u0 baseline '+str(p))
            filemap[str(p)]=digest(p);baseline_files.append(str(p))
    calibration=[]
    for c in protocol['configurations']:
        found=[]
        for p in sorted((source/'terminal/rollouts').glob('u*/*/RUN_COMPLETE.json')):
            r=read_json(p)
            if r['configuration']!=c:continue
            if not r['completed'] or r['phase']!='training' or not r['stochastic'] or r['plan_sha256']!=old['sha256']:
                raise ValueError('Invalid calibration trajectory')
            entry=next(e for e in old['entries'] if e['instance_id']==r['instance'])
            if entry['development_role']!='training':raise ValueError('Calibration must be training-only')
            trajectory=p.parent/'TRAJECTORY.npz'
            if digest(trajectory)!=r['trajectory_sha256']:raise ValueError('Training trajectory changed')
            found.append(trajectory);filemap[str(p)]=digest(p);break
        if not found:raise ValueError('No existing training candidate states for '+c)
        calibration.extend(found);filemap[str(found[0])]=digest(found[0])
    plan=dict(settings=s,pilot=protocol,source=str(source),source_plan_sha256=old['sha256'],shared_model=str(shared),
        entries=entries,training=train,tuning=tune,confirmation=confirmation,calibration=[str(p) for p in calibration],
        validation_baseline_files=baseline_files,normalization=old['normalization'],files=filemap)
    plan['sha256']=canonical_digest(plan);freeze(root/'PLAN.json',plan);return plan


def make_task(plan,entry,seed,model,phase,sample=False,record=False,update_number=0):
    return dict(instance=entry['instance_id'],seed=seed,arm='ICM_TERMINAL_PPO',phase=phase,sample=sample,
        model=str(model.resolve()),bundle_sha256=digest(model/'BUNDLE.json'),normalization=plan['normalization'],
        record=record,update=update_number,role=entry['development_role'])


def validate_task(root,plan,t,folder):
    entry=next((r for r in plan['entries'] if r['instance_id']==t['instance']),None)
    if entry is None or entry['development_role']!=t['role']:raise ValueError('Task split changed')
    if t['phase'] in ('training','training_reference'):
        u=t['update']
        if entry['development_role']!='training' or not 1<=u<=plan['pilot']['updates']:raise ValueError('Invalid training role/update')
        expected_entry=sorted([r for r in plan['training'] if r['configuration']==entry['configuration']],key=lambda r:r['instance_id'])[u-1]
        seed=plan['pilot']['training_seed_base']+(u-1)*len(plan['pilot']['configurations'])+plan['pilot']['configurations'].index(entry['configuration'])
        if entry!=expected_entry or t['seed']!=seed or not 1<=u<=plan['pilot']['updates']:raise ValueError('Unscheduled training episode')
        training=t['phase']=='training';number=u-1 if training else 0
        expected=root/'runs'/t['phase']/f'u{u:03d}'/entry['instance_id']/f'seed_{seed}'
        sample=record=training
    elif t['phase'] in ('tuning','confirmation','reproduction'):
        if entry['development_role']!='validation' or t['seed'] not in plan['settings']['ppo']['validation_seeds']:raise ValueError('Invalid validation task')
        group=plan['confirmation'] if t['phase']=='confirmation' else plan['tuning']
        if entry not in group:raise ValueError('Wrong validation subset')
        number=t['update'];record=False;sample=t['sample']
        if t['phase'] in ('tuning','reproduction') and sample:raise ValueError('Tuning/reproduction are greedy')
        if t['phase']=='reproduction' and number!=0:raise ValueError('Replay must be u0')
        if t['phase']=='confirmation':
            selected=read_json(root/'SELECTION.json')['best_updated_checkpoint']
            if number not in (0,selected) or (number==0 and not sample):raise ValueError('Unexpected confirmation model')
        expected=root/'runs'/t['phase']/f'u{number:03d}'/('sampled' if sample else 'greedy')/entry['instance_id']/f'seed_{t["seed"]}'
    else:raise ValueError('Unknown phase')
    model=root/'checkpoints'/f'update_{number:03d}'
    if not 0<=number<=plan['pilot']['updates'] or make_task(plan,entry,t['seed'],model,t['phase'],sample,record,t['update'])!=t:
        raise ValueError('Task/model differs from frozen schedule')
    if expected.resolve()!=folder.resolve():raise ValueError('Unexpected run folder')
    return entry


def run_worker(root,plan,t,out):
    validate_task(root,plan,t,out);verify(plan['files'])
    instance=root/'instances'/(t['instance']+'.json');info,_=read_instance(instance);s=plan['settings']
    learned=PilotPolicy(Path(t['model']),sample=t['sample'],seed=t['seed']+7000000,record=t['record']);holders=[]
    def factory(*args,**kwargs):
        ctl=BudgetController(*args,settings=s,arm=t['arm'],learned=learned,**kwargs);holders.append(ctl);return ctl
    torch.set_num_threads(1);normalizer=legacy_learning.normalizer(plan['normalization'])
    prior=base.clear_inherited_evaluation_cache;base.clear_inherited_evaluation_cache=lambda *args:None
    try:
        r=run_episode(EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',instance,out,None,normalizer,
            algorithm_seed=t['seed'],evaluation_budget=s['budget'],pop_size=s['population'],generations=1000000,
            evaluation_mode='ccp',component='BASE',scenario_count=s['scenario_count'],ccp_confidence=s['confidence'],
            uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=t['arm'])
    finally:base.clear_inherited_evaluation_cache=prior
    if not s['budget']-1<=r['real_objective_evaluation_count']<=s['budget'] or r['hidden_candidate_objective_evaluations']:
        raise ValueError('Wrong objective budget')
    ctl=holders[0];front=[list(_optimisation_vector(p)) for p in read_json(out/'final_archive_lineage.json')]
    snapshots=[p for p in ctl.budget_fronts if p['evaluation']!=s['budget']]
    snapshots.append(dict(evaluation=s['budget'],actual_evaluations=r['real_objective_evaluation_count'],hv=r['final_hypervolume'],
        front_raw=front,controller_seconds=time.perf_counter()-ctl.start,definition='actual final feasible nondominated front'))
    if {p['evaluation'] for p in snapshots}!=set(s['milestones']):raise ValueError('Missing checkpoints')
    trajectory=None
    if t['record']:learned.save_events(out/'TRAJECTORY.npz');trajectory=digest(out/'TRAJECTORY.npz')
    r.update(instance=info['instance_id'],configuration=info['configuration'],instance_digest=info['instance_digest'],seed=t['seed'],
        arm=t['arm'],phase=t['phase'],stochastic=t['sample'],plan_sha256=plan['sha256'],task_sha256=canonical_digest(t),
        model_bundle_sha256=t['bundle_sha256'],trajectory_sha256=trajectory,final_front_raw=front,
        scenario_digest=read_json(out/'configuration.json')['scenario_digest'],budget_fronts=snapshots,
        hostname=socket.gethostname(),peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2))
    write_json(out/'RUN_COMPLETE.json',r)


def jobs(root,plan,entries,number,phase,sampled=False):
    result=[];model=root/'checkpoints'/f'update_{number:03d}'
    for seed in plan['settings']['ppo']['validation_seeds']:
        for entry in entries:
            t=make_task(plan,entry,seed,model,phase,sampled,False,number)
            folder=root/'runs'/phase/f'u{number:03d}'/('sampled' if sampled else 'greedy')/entry['instance_id']/f'seed_{seed}'
            result.append((t,folder))
    return result


def cached_baselines(plan,entries):
    ids={r['instance_id'] for r in entries};rows=[]
    for p in plan['validation_baseline_files']:
        r=read_json(Path(p))
        if r['instance'] in ids:rows.append(r)
    return rows


def equal_mean(rows,key):
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:by[r['configuration']][r['instance']].append(r[key])
    return float(np.mean([np.mean([np.mean(v) for v in instances.values()]) for instances in by.values()]))


def paired(rows,reference,field='final_hypervolume',draws=3000):
    ref={(r['instance'],r['seed']):r for r in reference};pairs=[]
    if len(ref)!=len(reference) or len(rows)!=len(reference):raise ValueError('Incomplete reference panel')
    if {(r['instance'],r['seed']) for r in rows}!=set(ref):raise ValueError('Pair keys differ')
    for r in sorted(rows,key=lambda x:(x['configuration'],x['instance'],x['seed'])):
        b=ref[r['instance'],r['seed']];check_pair(r,b);v=float(r[field]);basehv=float(b[field])
        pairs.append(dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],own=v,reference=basehv,
            absolute=v-basehv,percent=100*(v-basehv)/basehv if basehv else None))
    rng=np.random.default_rng(140300000);by=defaultdict(lambda:defaultdict(list))
    for p in pairs:by[p['configuration']][p['instance']].append(p)
    deltas=[]
    for _ in range(draws):
        cfg=[]
        for group in by.values():
            ids=sorted(group);v=[]
            for i in rng.integers(0,len(ids),len(ids)):
                values=group[ids[int(i)]];v.append(np.mean([values[int(j)]['absolute'] for j in rng.integers(0,len(values),len(values))]))
            cfg.append(np.mean(v))
        deltas.append(np.mean(cfg))
    return dict(pairs=len(pairs),mean_hv=equal_mean(pairs,'own'),mean_absolute_difference=equal_mean(pairs,'absolute'),
        mean_paired_percent=equal_mean(pairs,'percent') if all(p['percent'] is not None for p in pairs) else None,
        absolute_ci_low=float(np.quantile(deltas,.025)),absolute_ci_high=float(np.quantile(deltas,.975)),
        wins=sum(p['absolute']>1e-12 for p in pairs),ties=sum(abs(p['absolute'])<=1e-12 for p in pairs),
        losses=sum(p['absolute']< -1e-12 for p in pairs),per_configuration={c:equal_mean([p for p in pairs if p['configuration']==c],'absolute') for c in by},
        zero_reference_pairs=sum(p['percent'] is None for p in pairs),per_run=pairs)


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(groups,destination,qualification,union=False):
    """No outcome filtering. Frozen normalization stays the selection criterion."""
    destination.mkdir(parents=True,exist_ok=True);summary=[];raw=[];report_json={}
    for name,rows in groups.items():
        if name=='共享监督_最高分':continue
        references=['共享监督_最高分']
        if name=='PPO_概率采样':references.append('共享监督_概率采样')
        for reference_name in references:
            for budget in (400,1000,5000):
                def at_budget(records):
                    result=[]
                    for r in records:
                        p=next(x for x in r['budget_fronts'] if x['evaluation']==budget)
                        result.append(dict(r,metric=p['hv']))
                    return result
                stats=paired(at_budget(rows),at_budget(groups[reference_name]),field='metric')
                summary.append({'评价次数':budget,'方法':name,'参照':reference_name,'运行对数':stats['pairs'],
                    ('平均HV_实例联合归一化_参考点1.1' if union else '平均HV_冻结训练归一化'):stats['mean_hv'],'平均配对提升(%)':stats['mean_paired_percent'],
                    '平均HV绝对差':stats['mean_absolute_difference'],'绝对差95%下限':stats['absolute_ci_low'],
                    '绝对差95%上限':stats['absolute_ci_high'],'胜':stats['wins'],'平':stats['ties'],'负':stats['losses'],
                    '平均B5000搜索秒':float(np.mean([r['optimisation_seconds'] for r in rows])) if budget==5000 else None})
                report_json[f'{name}/{reference_name}/{budget}']=stats
    for name,rows in groups.items():
        for r in rows:raw.append(dict(method=name,instance=r['instance'],configuration=r['configuration'],seed=r['seed'],
            final_hypervolume=r['final_hypervolume'],optimisation_seconds=r['optimisation_seconds'],
            instance_digest=r['instance_digest'],scenario_digest=r['scenario_digest'],model_bundle_sha256=r['model_bundle_sha256']))
    filename='00_配对结果_实例联合归一化.csv' if union else '00_配对结果_冻结训练归一化.csv'
    csv_write(destination/filename,summary);csv_write(destination/'PER_RUN.csv',raw)
    write_json(destination/'QUICK_RESULT.json',dict(completed=True,qualification=qualification,comparisons=report_json,
        new_evaluations_by_reporting=0,reference='shared supervised, not BASE',
        normalization=('per-instance cross-method/cross-seed/cross-budget min/max, reference 1.1; no volume division'
                       if union else 'unchanged frozen training normalization'),
        interval_scope='configuration equal; instances then seeds within fixed configurations; small development subset'))
    print('[PPO-PILOT] REPORT '+str(destination),flush=True)
    for r in summary:
        if r['评价次数']==5000:print('[PPO-PILOT] '+json.dumps(r,ensure_ascii=False),flush=True)


def union_report(groups,destination):
    """Post-hoc display only; never used for model fitting or checkpoint selection."""
    converted=copy.deepcopy(groups);bounds={}
    instances={r['instance'] for rows in converted.values() for r in rows}
    for instance in instances:
        records=[r for rows in converted.values() for r in rows if r['instance']==instance]
        points=[p for r in records for b in r['budget_fronts'] for p in b['front_raw']]
        if points:low=np.min(points,0);high=np.max(points,0);span=high-low
        else:low=np.zeros(3);high=np.zeros(3);span=np.zeros(3)
        bounds[instance]=dict(minimum=low.tolist(),maximum=high.tolist(),reference=[1.1]*3,budgets=[400,1000,5000])
        for r in records:
            for b in r['budget_fronts']:
                normalized=[[(v-low[j])/span[j] if span[j]>0 else 0. for j,v in enumerate(p)] for p in b['front_raw']]
                b['hv']=hv.hv3(normalized)
            r['final_hypervolume']=next(b['hv'] for b in r['budget_fronts'] if b['evaluation']==5000)
    report(converted,destination,'Post-hoc union-normalized confirmation; no training/selection use. HV is raw at reference (1.1,1.1,1.1), NOT divided by 1.331.',union=True)
    write_json(destination/'INSTANCE_BOUNDS.json',bounds)


def execute(a):
    root=a.out.resolve();source=a.source.resolve()
    if root==source or source in root.parents or root in source.parents:raise ValueError('Separate output required')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Pilot is already running')
        def stop(signum,frame):raise scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,source);p=plan['pilot'];first=root/'checkpoints/update_000'
            if not first.exists():
                temperatures,calibration=calibrate(Path(plan['shared_model']),[Path(x) for x in plan['calibration']],p)
                policy=PilotPolicy(Path(plan['shared_model']));policy.bundle['pilot']={**p,'temperatures':temperatures}
                write_json(root/'CALIBRATION.json',calibration);snapshot(policy,first,metrics=dict(update=0,calibration=calibration))
            print('[PPO-PILOT] frozen development pilot; first preview after update 1; no final test',flush=True)
            baseline=cached_baselines(plan,plan['tuning']);baseline_map={(r['instance'],r['seed']):r for r in baseline}
            replays=scheduler.execute_jobs(root,plan,jobs(root,plan,plan['tuning'],0,'reproduction')[:2],a.workers,'pilot/u0_reproduction')
            for r in replays:
                b=baseline_map[r['instance'],r['seed']];check_pair(r,b)
                for key in ('final_front_raw','final_hypervolume','evaluation_order_digest','real_objective_evaluation_count','selection'):
                    if r[key]!=b[key]:raise ValueError('u0 does not exactly reproduce source: '+key)
                if [(x['evaluation'],x['front_raw']) for x in r['budget_fronts']]!=[(x['evaluation'],x['front_raw']) for x in b['budget_fronts']]:
                    raise ValueError('u0 snapshots changed')
            tuning=[dict(update=0,mean_difference_vs_shared=0.)]
            for u in range(1,p['updates']+1):
                checkpoint=root/'checkpoints'/f'update_{u:03d}';previous=root/'checkpoints'/f'update_{u-1:03d}'
                pair_jobs=[];lookup={}
                for i,c in enumerate(p['configurations']):
                    entry=sorted([r for r in plan['training'] if r['configuration']==c],key=lambda r:r['instance_id'])[u-1]
                    seed=p['training_seed_base']+(u-1)*len(p['configurations'])+i
                    for phase,model,sample in [('training_reference',first,False),('training',previous,True)]:
                        t=make_task(plan,entry,seed,model,phase,sample,sample,u);folder=root/'runs'/phase/f'u{u:03d}'/entry['instance_id']/f'seed_{seed}'
                        pair_jobs.append((t,folder));lookup[entry['instance_id'],phase]=folder
                pair_rows=scheduler.execute_jobs(root,plan,pair_jobs,a.workers,f'pilot/train_pairs/u{u:03d}')
                indexed={(r['instance'],r['phase']):r for r in pair_rows};episodes=[];rewards=[]
                for instance in sorted({r['instance'] for r in pair_rows}):
                    r=indexed[instance,'training'];b=indexed[instance,'training_reference'];check_pair(r,b)
                    delta=r['final_hypervolume']-b['final_hypervolume'];episodes.append((lookup[instance,'training']/'TRAJECTORY.npz',delta))
                    rewards.append(dict(instance=instance,seed=r['seed'],policy_hv=r['final_hypervolume'],reference_hv=b['final_hypervolume'],paired_return=delta))
                if not checkpoint.exists():
                    policy=PilotPolicy(previous);reference=PilotPolicy(first);optimizer=optimizer_for(policy,p)
                    if (previous/'optimizer.pt').exists():
                        if digest(previous/'optimizer.pt')!=policy.bundle['optimizer_sha256']:raise ValueError('Optimizer changed')
                        optimizer.load_state_dict(load_checkpoint(previous/'optimizer.pt'))
                    write_json(root/'STATUS.json',dict(completed=False,stage=f'optimizer/u{u:03d}',pid=os.getpid()))
                    metrics=update(policy,reference,episodes,p,optimizer,p['optimizer_seed_base']+u)
                    metrics.update(update=u,reward_pairs=rewards);snapshot(policy,checkpoint,optimizer,metrics)
                    print('[PPO-PILOT] UPDATE '+json.dumps({k:metrics[k] for k in ('update','optimizer_steps','kl_early_stop','actor_parameter_l2_change')}),flush=True)
                tuning_jobs=jobs(root,plan,plan['tuning'],u,'tuning')
                early=scheduler.execute_jobs(root,plan,tuning_jobs[:len(plan['tuning'])],a.workers,f'pilot/preview/u{u:03d}')
                keys={(r['instance'],r['seed']) for r in early};early_reference=[r for r in baseline if (r['instance'],r['seed']) in keys]
                report({'共享监督_最高分':early_reference,f'PPO_u{u:03d}_最高分':early},root/'preview'/f'u{u:03d}',
                    'First fixed seed per configuration, descriptive only. Continue all predeclared tuning runs; no early outcome-based selection.')
                rows=scheduler.execute_jobs(root,plan,tuning_jobs,a.workers,f'pilot/tuning/u{u:03d}')
                score=paired(rows,baseline);tuning.append(dict(update=u,mean_difference_vs_shared=score['mean_absolute_difference'],statistics=score))
                report({'共享监督_最高分':baseline,f'PPO_u{u:03d}_最高分':rows},root/'tuning_reports'/f'u{u:03d}','Tuning results, all fixed pairs; reused u0 baseline.')
                write_json(root/'TRAINING_PROGRESS.json',dict(completed_update=u,validation=tuning))
            best=max(tuning,key=lambda r:(r['mean_difference_vs_shared'],-r['update']))
            updated=max(tuning[1:],key=lambda r:(r['mean_difference_vs_shared'],-r['update']))
            selection=dict(best_including_u0=best['update'],best_updated_checkpoint=updated['update'],tuning=tuning,
                selection=p['selection'],qualification='Best updated checkpoint is always reported and confirmed, even if u0 remains better. No alias substitutes for updated PPO.')
            freeze(root/'SELECTION.json',selection);u=updated['update']
            confirmations=jobs(root,plan,plan['confirmation'],u,'confirmation')+jobs(root,plan,plan['confirmation'],u,'confirmation',True)+jobs(root,plan,plan['confirmation'],0,'confirmation',True)
            completed=scheduler.execute_jobs(root,plan,confirmations,a.workers,'pilot/confirmation')
            selected_hash=digest(root/'checkpoints'/f'update_{u:03d}'/'BUNDLE.json')
            groups={'共享监督_最高分':cached_baselines(plan,plan['confirmation']),
                'PPO_最高分':[r for r in completed if not r['stochastic']],
                'PPO_概率采样':[r for r in completed if r['stochastic'] and r['model_bundle_sha256']==selected_hash],
                '共享监督_概率采样':[r for r in completed if r['stochastic'] and r['model_bundle_sha256']!=selected_hash]}
            report(groups,root/'results',p['qualification']);union_report(groups,root/'results_union')
            final=paired(groups['PPO_最高分'],groups['共享监督_最高分'])
            write_json(root/'STATUS.json',dict(completed=True,stage='all',selected_updated_checkpoint=u,best_tuning_including_u0=best['update'],
                confirmation_mean_hv_difference=final['mean_absolute_difference'],confirmation_direction_positive=final['mean_absolute_difference']>0,
                final_test_opened=False,automatic_further_experiments=False))
            print('[PPO-PILOT] COMPLETE '+json.dumps(read_json(root/'STATUS.json')),flush=True)
        except scheduler.Paused as e:
            write_json(root/'STATUS.json',dict(completed=False,paused=True,reason=str(e),resume='same command; completed jobs reused'))
            print('[PPO-PILOT] PAUSED '+str(e),flush=True)
        except Exception as e:
            write_json(root/'STATUS.json',dict(completed=False,failed=True,error=str(e)));raise


def main():
    diag=HERE.parent/'transformer_ppo_runs/diagnostics';parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=diag/'ccp_component_joint_b5000_v1')
    parser.add_argument('--out',type=Path,default=diag/'ccp_joint_ppo_b5000_pilot_v2')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--task',type=Path);parser.add_argument('--run-out',type=Path);args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if args.worker:
        plan=read_json(args.out/'PLAN.json')
        if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:raise ValueError('Plan changed')
        run_worker(args.out,plan,read_json(args.task),args.run_out)
    else:execute(args)


if __name__=='__main__':main()
