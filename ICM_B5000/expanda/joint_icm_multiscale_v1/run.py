#!/usr/bin/env python3
"""Audit -> historical shared pretraining -> full-budget collection -> shared fit."""
from bootstrap import *
import argparse
import fcntl
import os
import signal
import subprocess
import time


class Paused(Exception):pass


def prepare(root,source,settings):
    from data import audit_source
    instances,folders,files,report=audit_source(source,settings)
    if digest(ENGINE/'baseline_uncertainty.py')!=settings['baseline_file_sha256']:
        raise ValueError('Frozen transport evaluator changed')
    import platform,numpy,torch
    for info in instances:freeze(root/'instances'/(info['instance_id']+'.json'),info)
    files.update({str(root/'instances'/(r['instance_id']+'.json')):digest(root/'instances'/(r['instance_id']+'.json')) for r in instances})
    plan={'protocol':settings,'source':str(source),'files':{**runtime_files(),**files},
          'instances':[r['instance_id'] for r in instances],
          'source_folders':[str(p) for p in folders],
          'environment':{'python':platform.python_version(),'numpy':str(numpy.__version__),'torch':str(torch.__version__)}}
    plan['sha256']=canonical_digest(plan);freeze(root/'PLAN.json',plan)
    freeze(root/'instances/catalog.json',{'instances':instances,'source_sha256':digest(source/'instances/catalog.json'),
                                         'screen_instances_included':False})
    freeze(root/'SOURCE_AUDIT.json',report)
    print('[MULTI] 54 train / 18 validation, S0-S8; source B20000 labels verified; no screen/test data',flush=True)
    return plan,instances,folders


def execute_jobs(root,settings,jobs,sha,workers):
    running={};pending=list(jobs);done=0;last=0;start=time.monotonic()
    try:
        while pending or running:
            if time.monotonic()-start>settings['invocation_seconds']:raise Paused('Invocation time cap reached')
            while pending and len(running)<workers:
                name,seed,arm,folder=pending.pop(0)
                old=legacy_runner.completed(folder,sha)
                if old:
                    done+=1;print(f'[MULTI-COLLECT] reused {done}/{len(jobs)} {name}/{arm}',flush=True);continue
                if folder.exists():
                    archive=root/'interrupted';archive.mkdir(exist_ok=True)
                    folder.rename(archive/f'{name}_{arm}_{time.time_ns()}')
                folder.mkdir(parents=True)
                logfile=root/'logs'/f'{name}_{arm}_{time.time_ns()}.log';logfile.parent.mkdir(exist_ok=True)
                log=logfile.open('w')
                command=[sys.executable,'-u',str(HERE/'run.py'),'--worker','--out',str(root),
                         '--name',name,'--seed',str(seed),'--arm',arm,'--run-out',str(folder)]
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[str(folder)]=(proc,log,time.monotonic(),folder,logfile,name,arm)
                print(f'[MULTI-COLLECT] start {name}/{arm} PID={proc.pid} log={logfile}',flush=True)
            for key,(proc,log,t,folder,logfile,name,arm) in list(running.items()):
                if time.monotonic()-t>settings['task_timeout_seconds']:raise Paused('Worker time limit: '+str(logfile))
                if proc.poll() is None:continue
                log.close();del running[key]
                if proc.returncode:
                    with logfile.open('rb') as f:
                        f.seek(max(0,logfile.stat().st_size-8000));tail=f.read().decode(errors='replace')
                    raise RuntimeError(f'Worker exited {proc.returncode}: {logfile}\n{tail}')
                if not legacy_runner.completed(folder,sha):raise RuntimeError('No completion record: '+str(folder))
                done+=1;print(f'[MULTI-COLLECT] complete {done}/{len(jobs)} {name}/{arm}',flush=True)
            if time.monotonic()-last>=30:
                state={'completed':False,'stage':'collection','done':done,'total':len(jobs),
                       'running':[{'pid':p.pid,'instance':n,'arm':a,'seconds':int(time.monotonic()-t),'log':str(l)}
                                  for p,_,t,_,l,n,a in running.values()]}
                write_json(root/'STATUS.json',state)
                print(f'[MULTI] heartbeat {done}/{len(jobs)} running={len(running)} elapsed={time.monotonic()-start:.0f}s',flush=True)
                last=time.monotonic()
            if running:time.sleep(1)
    finally:
        for p,*_ in running.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p,log,*_ in running.values():
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            log.close()


def train_stage(root,folders,plan,preview):
    from data import load_dataset
    from training import fit
    settings=plan['protocol'];stage='pretrain' if preview else 'shared_model'
    write_json(root/'STATUS.json',{'completed':False,'stage':stage,'pid':os.getpid()})
    rows,norm=load_dataset(folders,settings['progress_denominator'])
    files={str(root/'PLAN.json'):digest(root/'PLAN.json'),**runtime_files()}
    for f in folders:
        for name in ('DATA.json','RUN_COMPLETE.json'):files[str(f/name)]=digest(f/name)
    coverage={'full_budget_ready':not preview,'trajectory_budget':20000 if preview else settings['budget'],
              'progress_denominator':settings['progress_denominator'],
              'fixed_initial_seeds_preserved':not preview,
              'use':'historical-label pretraining diagnostic' if preview else 'frozen supervised model for full-budget joint evaluation / future PPO',
              'old_pretrain_weights_used_for_full_fit':False}
    b=fit(rows,norm,settings,root/stage,files,coverage)
    if not preview:
        write_json(root/'SHARED_READY.json',{'completed':True,'model':str(root/stage),
            'bundle_sha256':digest(root/stage/'BUNDLE.json'),'full_budget_ready':True,
            'supervision_search_results_available':False,'ppo_started':False,
            'next_stage':'paired full-budget evaluation and terminal PPO; not executed by this supervision entry point'})
    return b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    diag=EXPANDA/'transformer_ppo_runs/diagnostics'
    p.add_argument('--source',type=Path,default=diag/'ccp_component_screen_b20_v1')
    p.add_argument('--out',type=Path,default=diag/'ccp_icm_multiscale_shared_v1')
    p.add_argument('--stage',choices=['all','audit','pretrain','collect','train'],default='all')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--name');p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--run-out',type=Path)
    a=p.parse_args();root=a.out.resolve();source=a.source.resolve()
    if root==source or source in root.parents or root in source.parents:raise ValueError('Use a separate output directory')
    root.mkdir(parents=True,exist_ok=True)
    if a.worker:
        plan=read_json(root/'PLAN.json')
        if canonical_digest({k:v for k,v in plan.items() if k!='sha256'})!=plan['sha256']:raise ValueError('Plan changed')
        # Runtime files only: immutable source feedback was audited before this stage.
        runtime=runtime_files()
        verify({p:s for p,s in plan['files'].items() if p in runtime or Path(p).parent==root/'instances'})
        info,_=read_instance(root/'instances'/(a.name+'.json'))
        if a.name not in plan['instances'] or a.arm not in ('COLLECT_I','COLLECT_CM') or a.seed!=info['algorithm_seed']:
            raise ValueError('Worker is outside frozen collection schedule')
        expected=root/'collection'/a.name/a.arm
        if a.run_out.resolve()!=expected:raise ValueError('Unexpected worker output')
        from collection import work
        work(root,plan['protocol'],a.name,a.seed,a.arm,expected,plan['sha256']);return
    if a.workers<1:raise ValueError('workers must be positive')
    with (root/'.pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This output already has an active pipeline')
        def stop(signum,frame):raise Paused(f'Received signal {signum}')
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try:
            settings=read_json(HERE/'protocol.json')
            plan,instances,folders=prepare(root,source,settings)
            if a.stage in ('all','pretrain'):train_stage(root,folders,plan,True)
            if a.stage in ('all','collect'):
                from collection import jobs
                execute_jobs(root,settings,jobs(root,instances),plan['sha256'],a.workers)
            if a.stage in ('all','train'):
                from collection import completed_dataset
                full=completed_dataset(root,settings,instances,plan['sha256'])
                train_stage(root,full,plan,False)
            ready=(root/'SHARED_READY.json').exists()
            write_json(root/'STATUS.json',{'completed':ready,'requested_stage_complete':True,'stage':a.stage,
                'shared_ready':ready,'ppo_started':False,'final_test_opened':False})
            print(f'[MULTI] requested stage complete; shared_ready={ready}; no PPO or test searches started',flush=True)
        except Paused as e:
            write_json(root/'STATUS.json',{'completed':False,'paused':True,'reason':str(e),'resume':'same command; completed jobs reused; training resumes at epoch boundary'})
            print('[MULTI] PAUSED '+str(e),flush=True)
        except Exception as e:
            write_json(root/'STATUS.json',{'completed':False,'failed':True,'error':str(e)})
            raise


if __name__=='__main__':main()
