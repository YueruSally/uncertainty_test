#!/usr/bin/env python3
"""Paired five-arm development study with bounded complete-episode joint PPO."""
from bridge import *
from prepare import prepare,training_schedule
import argparse
import fcntl
import os
import signal
import subprocess
import time
import copy


class Paused(Exception):pass


def task(plan,name,seed,arm,phase,model=None,sample=False):
    return {'instance':name,'seed':int(seed),'arm':arm,'phase':phase,'sample':sample,
            'model':str(model.resolve()) if model else None,'bundle_sha256':digest(model/'BUNDLE.json') if model else None,
            'normalization':plan['normalization']}


def validate_task(root,plan,t,folder):
    """Reject split leakage, changed seeds, models or arbitrary worker outputs."""
    s=plan['settings'];name=t['instance'];arm=t['arm'];phase=t['phase']
    entry=next((r for r in plan['entries']+plan['screen'] if r['instance_id']==name),None)
    if entry is None or t['normalization']!=plan['normalization'] or arm not in METHODS:
        raise ValueError('Invalid task definition')
    checkpoints=root/'terminal/checkpoints'
    if phase=='training':
        match=[r for r in training_schedule(plan) if r['instance']==name and r['seed']==t['seed']]
        if len(match)!=1 or not t['sample'] or arm!='ICM_TERMINAL_PPO' or entry['development_role']!='training':
            raise ValueError('Training task outside frozen schedule')
        u=match[0]['update'];model=checkpoints/f'update_{u-1:03d}'
        expected=root/'terminal/rollouts'/f'u{u:03d}'/name
    elif phase=='validation':
        if t['sample'] or arm!='ICM_TERMINAL_PPO' or t['seed']!=s['ppo']['validation_seed'] or entry['development_role']!='validation':
            raise ValueError('Validation task outside frozen schedule')
        models={str((checkpoints/f'update_{u:03d}').resolve()):u for u in s['ppo']['validation_checkpoints']}
        if t['model'] not in models:raise ValueError('Unscheduled validation checkpoint')
        u=models[t['model']];model=checkpoints/f'update_{u:03d}'
        expected=root/'terminal/validation'/f'u{u:03d}'/name
    elif phase=='screen':
        if t['sample'] or t['seed'] not in s['screen_seeds'] or entry['development_role']!='screen':
            raise ValueError('Screen task outside frozen schedule')
        selected=read_json(root/'terminal/SELECTED.json')
        if digest(Path(selected['model'])/'BUNDLE.json')!=selected['bundle_sha256']:raise ValueError('Selected model changed')
        model={'BASE':None,'ICM_HEURISTIC':None,'ICM_INDEPENDENT':root/'independent_model',
               'ICM_SHARED_SUPERVISED':Path(plan['source'])/'shared_model','ICM_TERMINAL_PPO':Path(selected['model'])}[arm]
        expected=root/'screen'/name/f'seed_{t["seed"]}'/arm
    else:raise ValueError('Unknown task phase')
    if t!=task(plan,name,t['seed'],arm,phase,model,t['sample']) or folder.resolve()!=expected.resolve():
        raise ValueError('Worker model, hash or output differs from frozen task')
    return entry


def valid_completion(folder,t,plan):
    path=folder/'RUN_COMPLETE.json'
    if not path.exists():return None
    r=read_json(path)
    if not r['completed'] or r['plan_sha256']!=plan['sha256'] or r['task_sha256']!=canonical_digest(t):raise ValueError('Completed run has different provenance: '+str(folder))
    if r['hidden_candidate_objective_evaluations'] or not plan['settings']['budget']-1<=r['real_objective_evaluation_count']<=plan['settings']['budget']:raise ValueError('Invalid completed search budget')
    if r.get('trajectory_sha256') and digest(folder/'TRAJECTORY.npz')!=r['trajectory_sha256']:raise ValueError('Trajectory changed')
    return r


def execute_jobs(root,plan,jobs,workers,stage):
    pending=list(jobs);running={};results=[];heartbeat=0.;start=time.monotonic()
    try:
        while pending or running:
            if time.monotonic()-start>plan['settings']['invocation_seconds']:raise Paused('Stage time cap')
            while pending and len(running)<workers:
                t,folder=pending.pop(0);prior=valid_completion(folder,t,plan)
                if prior:results.append(prior);print(f'[MULTI-JOINT] reused {stage} {len(results)}/{len(jobs)} {t["instance"]}/{t["arm"]}',flush=True);continue
                if folder.exists():
                    archive=root/'interrupted';archive.mkdir(exist_ok=True)
                    folder.rename(archive/(folder.name+'_'+str(time.time_ns())))
                folder.mkdir(parents=True)
                spec=root/'tasks'/(canonical_digest(t)+'.json');freeze(spec,t)
                logfile=root/'logs'/(canonical_digest(t)+f'_{time.time_ns()}.log');logfile.parent.mkdir(exist_ok=True)
                log=logfile.open('w')
                cmd=[sys.executable,'-u',str(HERE2/'run.py'),'--worker','--out',str(root),'--task',str(spec),'--run-out',str(folder)]
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[str(folder)]=(p,log,time.monotonic(),t,folder,logfile)
                print(f'[MULTI-JOINT] start {stage} {t["instance"]}/{t["arm"]} PID={p.pid} log={logfile}',flush=True)
            for key,(p,log,started,t,folder,logfile) in list(running.items()):
                if time.monotonic()-started>plan['settings']['task_timeout_seconds']:raise Paused('Worker timeout: '+str(logfile))
                if p.poll() is None:continue
                log.close();del running[key]
                if p.returncode:
                    with logfile.open('rb') as f:f.seek(max(0,logfile.stat().st_size-6000));tail=f.read().decode(errors='replace')
                    raise RuntimeError(f'Worker failed {p.returncode}: {logfile}\n{tail}')
                r=valid_completion(folder,t,plan)
                if r is None:raise ValueError('Worker did not finish')
                results.append(r)
                print(f'[MULTI-JOINT] complete {stage} {len(results)}/{len(jobs)} HV={r["final_hypervolume"]:.7g} seconds={r["optimisation_seconds"]:.1f}',flush=True)
            if time.monotonic()-heartbeat>=30:
                state={'completed':False,'stage':stage,'done':len(results),'total':len(jobs),
                       'running':[{'pid':p.pid,'instance':t['instance'],'arm':t['arm'],'seconds':int(time.monotonic()-started),'log':str(l)} for p,_,started,t,_,l in running.values()]}
                write_json(root/'STATUS.json',state);print(f'[MULTI-JOINT] heartbeat {stage} {len(results)}/{len(jobs)} running={len(running)}',flush=True)
                heartbeat=time.monotonic()
            if running:time.sleep(1)
        return results
    finally:
        for p,*_ in running.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,log,*_ in running.values():
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()


def validate_checkpoint(root,plan,checkpoint,number,workers):
    seed=plan['settings']['ppo']['validation_seed'];jobs=[]
    for r in plan['entries']:
        if r['development_role']=='validation':
            jobs.append((task(plan,r['instance_id'],seed,'ICM_TERMINAL_PPO','validation',checkpoint),root/'terminal/validation'/f'u{number:03d}'/r['instance_id']))
    rows=execute_jobs(root,plan,jobs,workers,f'validation/u{number:03d}')
    return sorted(rows,key=lambda r:r['instance'])


def validation_score(rows,initial,configurations):
    baseline={r['instance']:r for r in initial};by={c:[] for c in configurations}
    for r in rows:
        b=baseline[r['instance']]
        if r['seed']!=b['seed'] or r['scenario_digest']!=b['scenario_digest']:raise ValueError('Validation pairing changed')
        by[r['configuration']].append(r['final_hypervolume']-b['final_hypervolume'])
    if set(baseline)!={r['instance'] for r in rows} or any(not v for v in by.values()):raise ValueError('Incomplete validation')
    return float(np.mean([np.mean(v) for v in by.values()]))


def terminal(root,plan,workers):
    from policy import TerminalPolicy,optimizer_for,update,snapshot
    s=plan['settings']['ppo'];source=Path(plan['source'])/'shared_model';first=root/'terminal/checkpoints/update_000'
    if not first.exists():snapshot(TerminalPolicy(source),first,metrics={'update':0,'initialized_from':digest(source/'BUNDLE.json')})
    schedule=training_schedule(plan);freeze(root/'terminal/TRAINING_SCHEDULE.json',schedule)
    validation=[];history=[]
    for u in range(s['updates']+1):
        checkpoint=root/'terminal/checkpoints'/f'update_{u:03d}'
        if u:
            previous=root/'terminal/checkpoints'/f'update_{u-1:03d}';jobs=[]
            for r in schedule:
                if r['update']==u:
                    jobs.append((task(plan,r['instance'],r['seed'],'ICM_TERMINAL_PPO','training',previous,True),root/'terminal/rollouts'/f'u{u:03d}'/r['instance']))
            rows=execute_jobs(root,plan,jobs,workers,f'train/u{u:03d}')
            if not (checkpoint/'BUNDLE.json').exists():
                write_json(root/'STATUS.json',{'completed':False,'stage':f'optimizer/u{u:03d}','pid':os.getpid()})
                policy=TerminalPolicy(previous);optimizer=optimizer_for(policy,s)
                if (previous/'optimizer.pt').exists():
                    if digest(previous/'optimizer.pt')!=policy.bundle['optimizer_sha256']:raise ValueError('Optimizer changed')
                    optimizer.load_state_dict(load_checkpoint(previous/'optimizer.pt'))
                by_name={r['instance']:r for r in rows};episodes=[(folder/'TRAJECTORY.npz',by_name[t['instance']]['final_hypervolume']) for t,folder in jobs]
                started=time.monotonic();metrics=update(policy,episodes,s,optimizer,s['optimizer_seed_base']+u)
                metrics.update(update=u,optimizer_seconds=time.monotonic()-started)
                snapshot(policy,checkpoint,optimizer,metrics)
            history.append(read_json(checkpoint/'UPDATE.json'))
        if u in s['validation_checkpoints']:
            rows=validate_checkpoint(root,plan,checkpoint,u,workers)
            initial=validation[0]['runs'] if validation else rows
            score=validation_score(rows,initial,plan['settings']['configurations'])
            validation.append({'update':u,'mean_difference_vs_shared':score,'runs':rows})
            print(f'[MULTI-PPO] validation update={u} paired_mean_HV_delta={score:+.8g}',flush=True)
        write_json(root/'terminal/TRAINING_PROGRESS.json',{'completed_update':u,'training':history,'validation':validation})
    selected=max(validation,key=lambda x:(x['mean_difference_vs_shared'],-x['update']))
    checkpoint=root/'terminal/checkpoints'/f'update_{selected["update"]:03d}'
    result={'update':selected['update'],'model':str(checkpoint),'bundle_sha256':digest(checkpoint/'BUNDLE.json'),
            'validation_mean_difference_vs_shared':selected['mean_difference_vs_shared'],
            'selection':'configuration-equal paired final HV delta; update 0 allowed; ties earlier',
            'training_searches':len(schedule),'validation_searches':sum(len(x['runs']) for x in validation)}
    freeze(root/'terminal/SELECTED.json',result);return result


def screen(root,plan,workers):
    selected=read_json(root/'terminal/SELECTED.json');ppo=Path(selected['model'])
    if digest(ppo/'BUNDLE.json')!=selected['bundle_sha256']:raise ValueError('Selected checkpoint changed')
    model={'BASE':None,'ICM_HEURISTIC':None,'ICM_INDEPENDENT':root/'independent_model',
           'ICM_SHARED_SUPERVISED':Path(plan['source'])/'shared_model','ICM_TERMINAL_PPO':ppo}
    jobs=[(task(plan,r['instance_id'],seed,m,'screen',model[m]),root/'screen'/r['instance_id']/f'seed_{seed}'/m)
          for r in plan['screen'] for seed in plan['settings']['screen_seeds'] for m in METHODS]
    rows=execute_jobs(root,plan,jobs,workers,'screen')
    from reporting import report
    result=report(rows,plan,root/'results',selected)
    costs={}
    for phase,folder in [('training','terminal/rollouts'),('validation','terminal/validation'),('screen','screen')]:
        records=[read_json(p) for p in (root/folder).rglob('RUN_COMPLETE.json')]
        costs[phase]={'completed_searches':len(records),'objective_evaluations':sum(r['real_objective_evaluation_count'] for r in records),
                      'optimization_seconds_sum':sum(r['optimisation_seconds'] for r in records)}
    result['costs']=costs
    result['interrupted_attempts']=len(list((root/'interrupted').iterdir())) if (root/'interrupted').exists() else 0
    result['cost_qualification']='Completed searches only; interrupted attempts, source supervision and optimizer time excluded. Interrupted logs are preserved.'
    write_json(root/'results/QUICK_RESULT.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);diag=EXPANDA/'transformer_ppo_runs/diagnostics'
    p.add_argument('--source',type=Path,default=diag/'ccp_icm_multiscale_shared_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_icm_multiscale_joint_v1')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--stage',choices=['all','prepare','independent','ppo','screen'],default='all')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--task',type=Path);p.add_argument('--run-out',type=Path)
    a=p.parse_args();root=a.out.resolve();source=a.source.resolve();torch.set_num_threads(1)
    if root==source or root in source.parents or source in root.parents:raise ValueError('Separate output required')
    root.mkdir(parents=True,exist_ok=True)
    if a.worker:
        plan=read_json(root/'PLAN.json');t=read_json(a.task)
        if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:raise ValueError('Plan changed')
        verify(plan['files'])
        entry=validate_task(root,plan,t,a.run_out)
        from search import run_search
        run_search(root/'instances'/(entry['instance_id']+'.json'),plan['settings'],t,a.run_out,plan['sha256']);return
    if a.workers<1:raise ValueError('workers must be positive')
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This experiment already has an active pipeline')
        def stop(signum,frame):raise Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            plan=prepare(root,source,read_json(HERE2/'protocol.json'))
            if a.stage in ('all','independent'):
                write_json(root/'STATUS.json',{'completed':False,'stage':'independent_supervision','pid':os.getpid()})
                fit_independent(source,root/'independent_model')
            if a.stage in ('all','ppo'):terminal(root,plan,a.workers)
            if a.stage in ('all','screen'):screen(root,plan,a.workers)
            finished=(root/'results/QUICK_RESULT.json').exists()
            write_json(root/'STATUS.json',{'completed':finished,'requested_stage_complete':True,'stage':a.stage,'final_test_opened':False})
            print(f'[MULTI-JOINT] COMPLETE requested_stage={a.stage} full_experiment={finished}',flush=True)
        except Paused as e:
            write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'same command; completed runs and updates reused'})
            print('[MULTI-JOINT] PAUSED '+str(e),flush=True)
        except Exception as e:
            write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)})
            raise


if __name__=='__main__':main()
