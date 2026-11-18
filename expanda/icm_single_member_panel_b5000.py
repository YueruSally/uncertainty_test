#!/usr/bin/env python3
"""Single-member deployment: paired PPO first, then the full 11-method development panel."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from statistics import mean, stdev

import icm_single_member_b5000 as previous

EXP = Path(__file__).resolve().parent
DIAG = EXP / 'transformer_ppo_runs/diagnostics'
OUT = DIAG / 'ccp_icm_single_member_panel_b5000_v1'
OLD = DIAG / 'ccp_component_joint_b5000_v1'
SCREEN = DIAG / 'ccp_joint_ppo_b5000_screen27_v1'
SHARED_ROOT = DIAG / 'ccp_icm_shared_single_991001_b5000_v1'
SHARED = previous.SHARED
PPO = 'ICM_TERMINAL_PPO'
INDEPENDENT = ('I_LEARN', 'C_LEARN', 'M_LEARN', 'ICM_INDEPENDENT')
RULES = ('BASE', 'I_RULE', 'C_RULE', 'M_RULE', 'ICM_HEURISTIC')
NAMES = dict(BASE='Base', I_RULE='I 启发式', I_LEARN='I Transformer',
             C_RULE='C 启发式', C_LEARN='C Transformer', M_RULE='M 启发式', M_LEARN='M Transformer',
             ICM_HEURISTIC='ICM 启发式', ICM_INDEPENDENT='ICM 独立 Transformer',
             ICM_SHARED_SUPERVISED='ICM 共享监督 Transformer', ICM_TERMINAL_PPO='ICM 终局联合 PPO u1')
SEED = previous.TRAIN_SEED
BUDGETS = previous.BUDGETS


def weight_name(bundle):
    name = f"seed_{SEED}_epoch_{bundle['selected']['epoch']}.pt"
    if len(bundle['checkpoints']) != 3 or bundle['checkpoints'].count(name) != 1:
        raise ValueError('Cannot identify original supervised member ' + name)
    return name


def make_single(s, source, name, dest):
    bundle = s.read(source/'BUNDLE.json')
    if len(bundle['checkpoints']) != 3 or name not in bundle['checkpoints']:
        raise ValueError('Expected three members including ' + name)
    sha = bundle['checkpoint_sha256'][name]
    if s.digest(source/name) != sha:
        raise ValueError('Source member changed')
    dest.mkdir(exist_ok=True, parents=True)
    target = dest/name
    if not target.exists():
        tmp = target.with_suffix('.tmp')
        shutil.copyfile(source/name, tmp)
        tmp.replace(target)
    if s.digest(target) != sha:
        raise ValueError(f'Changed weight {target}')
    s.freeze(dest/'BUNDLE.json', dict(bundle, checkpoints=[name], checkpoint_sha256={name:sha}))
    return s.digest(dest/'BUNDLE.json')


def ppo_member(s, shared, shr_name, trained):
    # snapshot() names PPO members by position. Verify u0 actor tensors against the
    # actual supervised weights before using that position in the updated bundle.
    from shared import load_checkpoint
    bundle = s.read(shared/'BUNDLE.json')
    index = bundle['checkpoints'].index(shr_name)
    name = f'member_{index}.pt'
    start = trained.parent/'update_000'
    for folder in (start, trained):
        b = s.read(folder/'BUNDLE.json')
        if b['checkpoints'] != [f'member_{i}.pt' for i in range(3)]:
            raise ValueError('Unexpected PPO member order')
    state = load_checkpoint(shared/shr_name)['model']
    initial = load_checkpoint(start/name)['model']
    if (set(initial) - set(state) != {'value_head.weight', 'value_head.bias'}
            or not all(k in initial and s.p.torch.equal(v, initial[k]) for k,v in state.items())):
        raise ValueError('u0 actor does not match the supervised training seed')
    return name


def old_file(method, name, seed):
    if method == SHARED:
        return previous.job_path(SHARED_ROOT, name, seed)/'RUN_COMPLETE.json'
    return OLD/'screen'/name/f'seed_{seed}'/method/'RUN_COMPLETE.json'


def new_folder(method, name, seed):
    return OUT/'runs'/name/f'seed_{seed}'/method


def prepare(s):
    old = s.checked_plan(OLD)
    screen = s.checked_plan(SCREEN)
    earlier = s.read(SHARED_ROOT/'PLAN.json')
    settings = old['settings']
    if (old['sha256'] != screen['original_plan_sha256'] or old['sha256'] != earlier['original_plan']
            or screen['selected_update'] != 1 or not s.read(SCREEN/'STATUS.json')['completed']
            or earlier['training_seed'] != SEED or screen['normalization'] != old['normalization']):
        raise ValueError('Mismatched development comparisons')
    if (settings['screen_seeds'] != [993001,993002,993003] or tuple(settings['milestones']) != BUDGETS
            or sorted(r['configuration'] for r in old['screen']) != [f'S{i}' for i in range(9)]
            or (settings['population'],settings['budget'],settings['scenario_count'],settings['pool_size']) != (250,5000,100,8)):
        raise ValueError('Unexpected development panel')
    shared = Path(old['source'])/'shared_model'
    independent = Path(old['independent_model'])
    trained = Path(screen['model'])
    if s.digest(trained/'BUNDLE.json') != screen['model_bundle_sha256'] or trained.name != 'update_001':
        raise ValueError('PPO checkpoint differs from the original greedy u1 deployment')
    if s.digest(shared/'BUNDLE.json') != earlier['source_model']:
        raise ValueError('Shared controls differ')
    shared_bundle = s.read(shared/'BUNDLE.json')
    shared_name = weight_name(shared_bundle)
    member_bundle = s.read(SHARED_ROOT/'one_member/BUNDLE.json')
    if member_bundle['checkpoints'] != [shared_name] or member_bundle['checkpoint_sha256'][shared_name] != shared_bundle['checkpoint_sha256'][shared_name]:
        raise ValueError('Reused shared model is a different member')
    paths = {'independent':OUT/'models/independent', 'ppo':OUT/'models/ppo_u1'}
    hashes = {
        'independent':make_single(s, independent, weight_name(s.read(independent/'BUNDLE.json')), paths['independent']),
        'ppo':make_single(s, trained, ppo_member(s, shared, shared_name, trained), paths['ppo'])
    }
    source_hash = s.canonical(earlier)
    frozen_controls = {}
    for entry in old['screen']:
        name = entry['instance_id']
        if s.read(OLD/'instances'/(name+'.json')) != entry:
            raise ValueError('Development instance differs: '+name)
        for seed in settings['screen_seeds']:
            scenario = screen['scenarios'][f'{name}/{seed}']
            for arm in (*RULES, SHARED):
                path = old_file(arm,name,seed)
                row = s.read(path)
                s.audit.validate_run(row,entry,seed,settings,scenario)
                if (row['arm'] != arm or row['phase'] != 'screen' or row['stochastic'] or
                    row['plan_sha256'] != (source_hash if arm==SHARED else old['sha256']) or
                    row['model_bundle_sha256'] != (earlier['single_model'] if arm==SHARED else None)):
                    raise ValueError('Wrong control '+str(path))
                frozen_controls[str(path)] = s.digest(path)
    plan = dict(settings=settings, entries=old['screen'], scenarios=screen['scenarios'], normalization=old['normalization'],
                source_plan=old['sha256'], source_u1_screen=screen['sha256'], shared_single_plan=source_hash,
                models={k:str(v) for k,v in paths.items()}, hashes=hashes, reused=frozen_controls,
                source_code={str(f):s.digest(f) for f in (Path(__file__).resolve(),Path(previous.__file__).resolve())},
                seed=SEED, ppo_update=1, new_searches=135, shared_control_runs=27, rule_control_runs=135)
    plan['sha256'] = s.canonical(plan)
    s.freeze(OUT/'PLAN.json',plan)
    return plan


def run_worker(s, plan, method, name, seed):
    import search_b5000 as search
    if method not in (PPO,*INDEPENDENT) or seed not in plan['settings']['screen_seeds']:
        raise ValueError('Unknown job')
    entry = next(e for e in plan['entries'] if e['instance_id']==name)
    bundle_type = 'ppo' if method == PPO else 'independent'
    model = Path(plan['models'][bundle_type])
    task = s.scheduler.task(plan,name,seed,method,'screen',model)
    if task['bundle_sha256'] != plan['hashes'][bundle_type] or len(s.read(model/'BUNDLE.json')['checkpoints']) != 1:
        raise ValueError('Single-member model changed')
    instance = OLD/'instances'/(name+'.json')
    if s.read(instance) != entry:
        raise ValueError('Instance changed')

    def validate(row):
        s.audit.validate_run(row,entry,seed,plan['settings'],plan['scenarios'][f'{name}/{seed}'])
        if (row['task_sha256'] != s.canonical(task) or row['plan_sha256'] != plan['sha256']
                or row['model_bundle_sha256'] != task['bundle_sha256'] or row['arm'] != method
                or row['stochastic'] or row['phase'] != 'screen'):
            raise ValueError('Incorrect worker result')

    dest = new_folder(method,name,seed)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if dest.is_dir():
            validate(s.read(dest/'RUN_COMPLETE.json'))
            return
        tmp = Path(tempfile.mkdtemp(prefix=method+'.attempt-',dir=dest.parent))
        if method == PPO:
            # This process runs just one task. The v2 fixed-u1 greedy selector uses
            # the original float64 gate and tie handling on a single member.
            search.TerminalPolicy = s.p.PilotPolicy
        row = search.run_search(instance,plan['settings'],task,tmp,plan['sha256'])
        validate(row)
        tmp.rename(dest)


def run_stage(plan, methods, workers, offset):
    jobs = [(m,e['instance_id'],seed) for m in methods for e in plan['entries']
            for seed in plan['settings']['screen_seeds']]
    def launch(job):
        method,name,seed = job
        log = OUT/'logs'/f'{method}_{name}_{seed}.log'
        cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',method,name,str(seed)]
        with log.open('a') as f:
            result = subprocess.run(cmd,cwd=EXP,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{job} failed; see {log}')
        return job
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch,job) for job in jobs}
        try:
            while pending:
                finished,pending = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                if not finished:
                    print(f'[SINGLE] heartbeat {offset+done}/135',flush=True)
                for future in finished:
                    job = future.result()
                    done += 1
                    print(f'[SINGLE] {offset+done}/135',*job,flush=True)
        finally:
            for future in pending:
                future.cancel()


def report(s, plan, methods, dest):
    groups = {m:[] for m in methods}
    for e in plan['entries']:
        name = e['instance_id']
        for seed in plan['settings']['screen_seeds']:
            for m in methods:
                if m in RULES or m == SHARED:
                    file = old_file(m,name,seed)
                    if s.digest(file) != plan['reused'][str(file)]:
                        raise ValueError('Reused control changed')
                else:
                    file = new_folder(m,name,seed)/'RUN_COMPLETE.json'
                r = s.read(file)
                s.audit.validate_run(r,e,seed,plan['settings'],plan['scenarios'][f'{name}/{seed}'])
                groups[m].append(r)
    normalized,bounds = s.audit.union_normalize(groups)
    s.freeze(dest/'INSTANCE_BOUNDS.json',bounds)
    main,direct,details = [],[],[]
    for budget in BUDGETS:
        at = {m:[dict(r,metric=next(b['hv'] for b in r['budget_fronts'] if b['evaluation']==budget))
                 for r in runs] for m,runs in normalized.items()}
        for m,ref in [(m,'BASE') for m in methods]+[(PPO,SHARED)]:
            v = s.p.paired(at[m],at[ref],field='metric',draws=3000)
            lo,hi = s.p.hv.interval(v['per_run'],'percent',3000)
            row = dict(评价次数=budget,方法=NAMES[m],参照=NAMES[ref],配对数=v['pairs'],平均HV=v['mean_hv'],
                       平均配对提升百分比=v['mean_paired_percent'],提升区间下限=lo,提升区间上限=hi,
                       平均HV差=v['mean_absolute_difference'],HV差区间下限=v['absolute_ci_low'],HV差区间上限=v['absolute_ci_high'],
                       胜=v['wins'],平=v['ties'],负=v['losses'])
            for field,label in [('optimisation_seconds','搜索'),('inference_seconds','推理')]:
                values = [r[field] for r in at[m]]
                row[label+'平均秒'] = mean(values) if budget==5000 else None
                row[label+'标准差秒'] = stdev(values) if budget==5000 else None
            (main if ref=='BASE' else direct).append(row)
            timing = {(r['instance'],r['seed']):r for r in at[m]}
            for pair in v['per_run']:
                r = timing[pair['instance'],pair['seed']]
                details.append(dict(method=m,reference_method=ref,evaluation=budget,**pair,
                                    optimisation_seconds=r['optimisation_seconds'] if budget==5000 else None,
                                    inference_seconds=r['inference_seconds'] if budget==5000 else None))
    for file,rows in [('RESULTS.csv',main),('PPO_vs_SHARED.csv',direct),('PER_PAIR.csv',details)]:
        path = dest/file
        tmp = path.with_suffix('.tmp')
        s.p.csv_write(tmp,rows)
        tmp.replace(path)
    print('[SINGLE] Report ready:',dest/'RESULTS.csv',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--worker',nargs=3,help=argparse.SUPPRESS)
    args = p.parse_args()
    s = previous.load_module(EXP)
    if args.worker:
        plan = s.read(OUT/'PLAN.json')
        if s.canonical({k:v for k,v in plan.items() if k!='sha256'}) != plan['sha256']:
            raise ValueError('Plan checksum changed')
        s.p.verify(plan['source_code'])
        method,name,seed = args.worker
        run_worker(s,plan,method,name,int(seed))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('[SINGLE] Verifying frozen weights, controls and 27 paired instances/seeds',flush=True)
        plan = prepare(s)
        (OUT/'logs').mkdir(exist_ok=True)
        run_stage(plan,(PPO,),args.workers,0)
        report(s,plan,('BASE',SHARED,PPO),OUT/'early')
        run_stage(plan,INDEPENDENT,args.workers,27)
        report(s,plan,tuple(NAMES),OUT/'final')
        s.write(OUT/'STATUS.json',dict(completed=True,new_searches=135,reused_runs=162,
                normalization='per instance across all seeds, three budgets and all methods in each report; reference (1.1,1.1,1.1)',
                training_updates=0,final_test_opened=False))
        print('[SINGLE] COMPLETE',OUT/'final/RESULTS.csv',flush=True)


if __name__=='__main__':
    main()
