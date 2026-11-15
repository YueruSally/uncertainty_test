#!/usr/bin/env python3
"""Frozen B5000 panel plus bounded terminal PPO, isolated from completed studies."""
from pathlib import Path
import sys
B5_DIR=Path(__file__).resolve().parent
TERMINAL_DIR=B5_DIR.parent/'joint_icm_multiscale_terminal_v1'
sys.path.insert(0,str(TERMINAL_DIR))
from bridge import *
from search_b5000 import ARMS,SINGLES
import argparse
import importlib.util
import json
import fcntl
import os
import signal
import time
from collections import defaultdict

spec=importlib.util.spec_from_file_location('b5000_legacy_runner',TERMINAL_DIR/'run.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
legacy.HERE2=B5_DIR  # Worker launches this entry point; original files never change.
task=legacy.task


def model_for(plan,arm,root):
    if arm.endswith('_LEARN') or arm=='ICM_INDEPENDENT':return Path(plan['independent_model'])
    if arm=='ICM_SHARED_SUPERVISED':return Path(plan['source'])/'shared_model'
    if arm=='ICM_TERMINAL_PPO':
        selected=read_json(root/'terminal/SELECTED.json');model=Path(selected['model'])
        if digest(model/'BUNDLE.json')!=selected['bundle_sha256']:raise ValueError('Selected PPO model changed')
        return model
    return None


def prepare(root,original):
    old=read_json(original/'PLAN.json')
    if canonical_digest({k:v for k,v in old.items() if k!='sha256'})!=old['sha256']:raise ValueError('Source plan changed')
    verify(old['files'])
    protocol=read_json(B5_DIR/'protocol.json');s=copy.deepcopy(old['settings'])
    if s['population']!=250 or s['scenario_count']!=100 or s['progress_denominator']!=50000 or len(old['screen'])!=9:
        raise ValueError('Expected full multiscale frozen source')
    configs=['S'+str(i) for i in range(9)]
    if s['configurations']!=configs or sorted(r['configuration'] for r in old['screen'])!=configs:
        raise ValueError('Expected exactly one development instance per S0-S8 configuration')
    if len(s['screen_seeds'])!=3 or len(set(s['screen_seeds']))!=3:raise ValueError('Expected three distinct common seeds')
    entries=old['entries']+old['screen']
    if len({r['instance_id'] for r in entries})!=len(entries) or len({r['batch_signature'] for r in entries})!=len(entries):
        raise ValueError('Overlapping source instances')
    for cfg in configs:
        for role,count in [('training',6),('validation',2),('screen',1)]:
            if sum(r['configuration']==cfg and r['development_role']==role for r in entries)!=count:
                raise ValueError('Wrong source split: '+cfg+'/'+role)
    s.update(budget=protocol['budget'],milestones=protocol['milestones'],population=protocol['population'])
    s['ppo'].update(updates=protocol['ppo_updates'],episodes_per_update=protocol['episodes_per_update'],
                    validation_checkpoints=protocol['validation_checkpoints'],validation_seeds=protocol['validation_seeds'],
                    selection='18 validation instances x 3 seeds; configuration-equal paired final B5000 HV delta vs u0; ties earlier')
    s.update(protocol=protocol['protocol'],maximum_training_searches=32,maximum_validation_searches=162,
             maximum_screen_searches=297,maximum_completed_search_evaluations=protocol['new_objective_evaluations_maximum'],
             reporting=protocol['main_report'],scope=protocol['comparison'])
    source=Path(old['source']);independent=original/'independent_model'
    filemap={**files(),**{str(p.resolve()):digest(p) for p in B5_DIR.glob('*.py')},str(B5_DIR/'protocol.json'):digest(B5_DIR/'protocol.json'),
             str(original/'PLAN.json'):digest(original/'PLAN.json')}
    bundles=[]
    for folder,loader in ((source/'shared_model',SharedPolicy),(independent,IndependentPolicy)):
        loader(folder);b=read_json(folder/'BUNDLE.json');bundles.append(b)
        if b['normalization']!=old['normalization'] or b['settings']['progress_denominator']!=50000:raise ValueError('Incompatible frozen normalization/features')
        filemap[str(folder/'BUNDLE.json')]=digest(folder/'BUNDLE.json')
        filemap.update({str(folder/p):sha for p,sha in b['checkpoint_sha256'].items()})
    for key in ('normalization','scalers','target_scales','training_instances','validation_instances'):
        if bundles[0][key]!=bundles[1][key]:raise ValueError('Independent/shared control differs: '+key)
    for component in COMPONENTS:
        for role in ('training','validation'):
            if set(bundles[0][role+'_instances'][component])!={r['instance_id'] for r in old['entries'] if r['development_role']==role}:
                raise ValueError('Frozen models do not match source split')
    for entry in old['entries']+old['screen']:
        path=original/'instances'/(entry['instance_id']+'.json');info,_=read_instance(path)
        if info!=entry:raise ValueError('Source instance changed')
        freeze(root/'instances'/path.name,entry);filemap[str(root/'instances'/path.name)]=digest(root/'instances'/path.name)
    plan=dict(settings=s,source=str(source),independent_model=str(independent),source_experiment=str(original),
        source_plan_sha256=old['sha256'],entries=old['entries'],screen=old['screen'],normalization=old['normalization'],
        protocol=protocol,files=filemap)
    plan['sha256']=canonical_digest(plan);freeze(root/'PLAN.json',plan);return plan


def validate_task(root,plan,t,folder):
    entry=next((r for r in plan['entries']+plan['screen'] if r['instance_id']==t['instance']),None)
    if entry is None or t['arm'] not in ARMS:raise ValueError('Unknown instance/method')
    if t['phase']=='training':return legacy.validate_task(root,plan,t,folder)
    if t['phase']=='validation':
        models={str((root/'terminal/checkpoints'/f'update_{u:03d}').resolve()):u for u in plan['settings']['ppo']['validation_checkpoints']}
        if (t['model'] not in models or t['arm']!='ICM_TERMINAL_PPO' or t['sample'] or
            entry['development_role']!='validation' or t['seed'] not in plan['settings']['ppo']['validation_seeds']):
            raise ValueError('Unscheduled validation')
        model=Path(t['model']);u=models[t['model']]
        expected=root/'terminal/validation'/f'u{u:03d}'/t['instance']/f'seed_{t["seed"]}'
    elif t['phase']=='screen':
        if t['sample'] or entry['development_role']!='screen' or t['seed'] not in plan['settings']['screen_seeds']:raise ValueError('Invalid screen split/seed')
        model=model_for(plan,t['arm'],root);expected=root/'screen'/t['instance']/f'seed_{t["seed"]}'/t['arm']
    else:raise ValueError('Unknown phase')
    if t!=task(plan,t['instance'],t['seed'],t['arm'],t['phase'],model,t['sample']) or expected.resolve()!=folder.resolve():
        raise ValueError('Task or output differs from frozen plan')
    return entry


def validate_checkpoint(root,plan,checkpoint,number,workers):
    jobs=[(task(plan,r['instance_id'],seed,'ICM_TERMINAL_PPO','validation',checkpoint),
           root/'terminal/validation'/f'u{number:03d}'/r['instance_id']/f'seed_{seed}')
          for r in plan['entries'] if r['development_role']=='validation' for seed in plan['settings']['ppo']['validation_seeds']]
    rows=legacy.execute_jobs(root,plan,jobs,workers,f'validation/u{number:03d}')
    return sorted(rows,key=lambda r:(r['instance'],r['seed']))


def validation_score(rows,initial,configurations):
    baseline={(r['instance'],r['seed']):r for r in initial}
    if len(initial)!=len(baseline) or set(baseline)!={(r['instance'],r['seed']) for r in rows} or len(rows)!=len(baseline):raise ValueError('Incomplete paired validation')
    groups=defaultdict(lambda:defaultdict(list))
    for r in sorted(rows,key=lambda r:(r['configuration'],r['instance'],r['seed'])):
        b=baseline[r['instance'],r['seed']]
        if r['scenario_digest']!=b['scenario_digest'] or r['configuration']!=b['configuration']:raise ValueError('Unpaired scenarios/configurations')
        groups[r['configuration']][r['instance']].append(r['final_hypervolume']-b['final_hypervolume'])
    if set(groups)!=set(configurations):raise ValueError('Missing validation configuration')
    return float(np.mean([np.mean([np.mean(groups[c][i]) for i in sorted(groups[c])]) for c in configurations]))


legacy.validate_checkpoint=validate_checkpoint
legacy.validation_score=validation_score


def jobs_for(root,plan,arms):
    return [(task(plan,r['instance_id'],seed,m,'screen',model_for(plan,m,root)),root/'screen'/r['instance_id']/f'seed_{seed}'/m)
            for r in plan['screen'] for seed in plan['settings']['screen_seeds'] for m in arms]


def read_runs(root,plan,arms):
    rows=[]
    for t,f in jobs_for(root,plan,arms):
        r=legacy.valid_completion(f,t,plan)
        if r is not None:rows.append(r)
    return rows


def verify_reproduction(a,b):
    for key in ('final_front_raw','final_hypervolume','evaluation_order_digest','real_objective_evaluation_count',
                'instance_digest','scenario_digest','selection'):
        if a[key]!=b[key]:raise ValueError('u0 reproduction mismatch: '+key)
    if [(x['evaluation'],x['front_raw']) for x in a['budget_fronts']] != [(x['evaluation'],x['front_raw']) for x in b['budget_fronts']]:
        raise ValueError('u0 budget snapshots differ')


def screen_ppo(root,plan,workers):
    selected=read_json(root/'terminal/SELECTED.json');jobs=jobs_for(root,plan,['ICM_TERMINAL_PPO'])
    if selected['update']==0:
        jobs=jobs[:2]
        checks=legacy.execute_jobs(root,plan,jobs,workers,'screen/u0_reproduction')
        shared={(r['instance'],r['seed']):r for r in read_runs(root,plan,['ICM_SHARED_SUPERVISED'])}
        for r in checks:verify_reproduction(shared[r['instance'],r['seed']],r)
        write_json(root/'PPO_REUSE.json',dict(selected_update=0,alias_of='ICM_SHARED_SUPERVISED',
            actual_reproduction_runs=len(checks),reused_rows=len(plan['screen'])*len(plan['settings']['screen_seeds'])-len(checks),
            checked_keys=[dict(instance=r['instance'],seed=r['seed']) for r in checks]))
    else:legacy.execute_jobs(root,plan,jobs,workers,'screen/selected_ppo')


def estimate_cost(root,plan,rows,workers):
    per_method={m:[r['optimisation_seconds'] for r in rows if r['arm']==m] for m in ARMS[:-1]}
    means={m:float(np.mean(v)) for m,v in per_method.items() if v}
    count=len(plan['screen'])*len(plan['settings']['screen_seeds'])
    screen_work=sum(max(0,count-len(per_method[m]))*means[m] for m in means)
    p=plan['settings']['ppo']
    validation=sum(r['development_role']=='validation' for r in plan['entries'])*len(p['validation_seeds'])*len(p['validation_checkpoints'])
    caps={'terminal/rollouts':p['updates']*p['episodes_per_update'],'terminal/validation':validation}
    remaining=sum(max(0,cap-len(list((root/folder).rglob('RUN_COMPLETE.json')))) for folder,cap in caps.items())+count
    ppo_work=remaining*means['ICM_SHARED_SUPERVISED']
    estimate=dict(observed_searches=len(rows),workers=workers,seconds_per_method=means,
        remaining_frozen_panel_hours=screen_work/workers/3600,
        terminal_training_validation_screen_upper_schedule_hours=ppo_work/workers/3600,
        rough_remaining_search_hours=(screen_work+ppo_work)/workers/3600,
        qualification='Duration projection, not a bound: current observed configurations only; PPO assumed as costly as shared inference. Excludes optimizer, process/setup, reporting, parallel barriers and contention. Update 0 can save 25 final searches.')
    write_json(root/'COST_ESTIMATE.json',estimate)
    print('[B5000-COST] '+json.dumps(estimate),flush=True)


def execute(a):
    root=a.out.resolve();original=a.source.resolve()
    if root==original or original in root.parents or root in original.parents:raise ValueError('Separate output required')
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('B5000 pipeline already active')
        def stop(signum,frame):raise legacy.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,original)
            from reporting_b5000 import report
            deterministic=ARMS[:-1]
            if a.stage in ('all','screen'):
                jobs=jobs_for(root,plan,deterministic)
                # A predeclared first block provides a quick descriptive report.
                first=legacy.execute_jobs(root,plan,jobs[:len(deterministic)],a.workers,'screen/first_instance_seed')
                report(first,plan,root/'preview',partial=True)
                estimate_cost(root,plan,first,a.workers)
                print('[B5000] first-block preview ready; no outcome-based stopping',flush=True)
                legacy.execute_jobs(root,plan,jobs,a.workers,'screen/ten_frozen_methods')
                frozen=read_runs(root,plan,deterministic)
                report(frozen,plan,root/'supervised_results')
                estimate_cost(root,plan,frozen,a.workers)
            if a.stage in ('all','ppo'):legacy.terminal(root,plan,a.workers)
            if a.stage in ('all','ppo-screen'):
                screen_ppo(root,plan,a.workers)
                report(read_runs(root,plan,ARMS),plan,root/'results',reuse=read_json(root/'PPO_REUSE.json') if (root/'PPO_REUSE.json').exists() else None)
            finished=(root/'results/QUICK_RESULT.json').exists()
            write_json(root/'STATUS.json',dict(completed=finished,requested_stage_complete=True,stage=a.stage,final_test_opened=False))
            print('[B5000] COMPLETE requested stage; complete study='+str(finished),flush=True)
        except legacy.Paused as e:
            write_json(root/'STATUS.json',dict(completed=False,paused=True,reason=str(e),resume='same command; completed runs and updates reused'))
            print('[B5000] PAUSED '+str(e),flush=True)
        except Exception as e:
            write_json(root/'STATUS.json',dict(completed=False,failed=True,error=str(e)));raise


def main():
    diag=B5_DIR.parent/'transformer_ppo_runs/diagnostics'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=diag/'ccp_icm_multiscale_joint_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_component_joint_b5000_v1')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--stage',choices=['all','prepare','screen','ppo','ppo-screen'],default='all')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--task',type=Path);p.add_argument('--run-out',type=Path)
    a=p.parse_args()
    if a.workers<1:p.error('positive worker count required')
    if a.worker:
        plan=read_json(a.out/'PLAN.json');t=read_json(a.task)
        if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:raise ValueError('Plan changed')
        verify(plan['files']);entry=validate_task(a.out,plan,t,a.run_out)
        from search_b5000 import run_search
        run_search(a.out/'instances'/(entry['instance_id']+'.json'),plan['settings'],t,a.run_out,plan['sha256'])
    else:execute(a)


if __name__=='__main__':main()
