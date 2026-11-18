#!/usr/bin/env python3
"""Portable single-member inference. No training; supplied results stay read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import time

PKG = Path(__file__).resolve().parent
EXP = PKG / 'expanda'
ARMS = ('BASE', 'I_RULE', 'I_LEARN', 'C_RULE', 'C_LEARN', 'M_RULE', 'M_LEARN',
        'ICM_HEURISTIC', 'ICM_INDEPENDENT', 'ICM_SHARED_SUPERVISED', 'ICM_TERMINAL_PPO')
NAMES = ('Base', 'I 启发式', 'I Transformer', 'C 启发式', 'C Transformer',
         'M 启发式', 'M Transformer', 'ICM 启发式', 'ICM 独立 Transformer',
         'ICM 共享监督 Transformer', 'ICM 联合 PPO u1')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(p)


def sha(p):
    value = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(include_wheels=True):
    manifest = read(PKG / 'MANIFEST.json')
    for name, expected in manifest['files'].items():
        if name.startswith('wheelhouse/') and not include_wheels:
            continue
        p = PKG / name
        if not p.is_file() or sha(p) != expected:
            raise ValueError('Missing/changed package file: ' + name)
    plan = read(PKG / 'assets/PLAN.json')
    for kind in ('independent', 'shared', 'ppo'):
        b = read(PKG / 'models' / kind / 'BUNDLE.json')
        if len(b['checkpoints']) != 1:
            raise ValueError(kind + ' must contain exactly one member')
        name = b['checkpoints'][0]
        if sha(PKG / 'models' / kind / name) != b['checkpoint_sha256'][name]:
            raise ValueError('Changed model member: ' + kind)
    return plan


def runtime():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(EXP))
    import icm_single_member_b5000 as original
    return original.load_module(EXP)


def worker(plan, arm, instance, seed, out):
    s = runtime()
    import search_b5000 as search
    kind = ('shared' if arm == 'ICM_SHARED_SUPERVISED' else 'ppo' if arm == 'ICM_TERMINAL_PPO'
            else 'independent' if arm.endswith('_LEARN') or arm == 'ICM_INDEPENDENT' else None)
    model = PKG / 'models' / kind if kind else None
    task = s.scheduler.task(plan, instance, seed, arm, 'screen', model, False)
    if arm == 'ICM_TERMINAL_PPO':
        search.TerminalPolicy = s.p.PilotPolicy
    entry = next(e for e in plan['entries'] if e['instance_id'] == instance)
    dest = out / 'runs' / instance / ('seed_' + str(seed)) / arm
    if dest.exists():
        raise ValueError('Incomplete run directory already exists: ' + str(dest))
    row = search.run_search(PKG / 'assets/instances' / (instance + '.json'),
                            plan['settings'], task, dest, sha(PKG / 'assets/PLAN.json'))
    s.audit.validate_run(row, entry, seed, plan['settings'], plan['scenarios'][f'{instance}/{seed}'])
    if row['stochastic'] or row['arm'] != arm or row['model_bundle_sha256'] != task['bundle_sha256']:
        raise ValueError('Deployment differs from the frozen task')
    write(dest / 'PORTABLE_COMPLETE.json', dict(plan_sha=sha(PKG / 'assets/PLAN.json'),
                                               run_sha=sha(dest / 'RUN_COMPLETE.json')))


def report(plan, out, entries, seeds):
    s = runtime()
    groups = {arm: [read(out / 'runs' / e['instance_id'] / ('seed_' + str(seed)) / arm / 'RUN_COMPLETE.json')
                    for e in entries for seed in seeds] for arm in ARMS}
    normalized, bounds = s.audit.union_normalize(groups)
    write(out / 'INSTANCE_BOUNDS.json', bounds)
    rows = []
    for budget in (400, 1000, 5000):
        at = {arm: [dict(r, metric=next(x['hv'] for x in r['budget_fronts'] if x['evaluation'] == budget))
                    for r in records] for arm, records in normalized.items()}
        for arm, label in zip(ARMS, NAMES):
            v = s.p.paired(at[arm], at['BASE'], field='metric', draws=3000)
            row = dict(评价次数=budget, 方法=label, 参照='Base', 配对数=v['pairs'], 平均HV=v['mean_hv'],
                       相对Base平均配对提升百分比=v['mean_paired_percent'],
                       终点提升95下限='', 终点提升95上限='', 终点胜='', 终点平='', 终点负='',
                       终点平均搜索秒='', 终点搜索标准差秒='', 终点平均推理秒='')
            if budget == 5000:
                lo, hi = s.p.hv.interval(v['per_run'], 'percent', 3000)
                times = [r['optimisation_seconds'] for r in at[arm]]
                row.update(终点提升95下限=lo, 终点提升95上限=hi,
                           终点胜=v['wins'] if arm != 'BASE' else '—',
                           终点平=v['ties'] if arm != 'BASE' else '—',
                           终点负=v['losses'] if arm != 'BASE' else '—',
                           终点平均搜索秒=mean(times), 终点搜索标准差秒=stdev(times) if len(times) > 1 else '',
                           终点平均推理秒=mean(r['inference_seconds'] for r in at[arm]))
            rows.append(row)
    s.p.csv_write(out / 'COMPUTED_RESULTS.csv', rows)
    write(out / 'STATUS.json', dict(completed=True, pairs_per_method=len(entries)*len(seeds),
          searches=11*len(entries)*len(seeds), training_updates=0, source_table_unchanged=True,
          note='Only endpoint W/T/L is exported. Newly computed results are separate from the supplied summary.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Verify files and load all three single-member bundles; no search')
    p.add_argument('--quick', action='store_true', help='First instance and first seed, all 11 methods; separate report')
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    p.add_argument('--out', type=Path)
    p.add_argument('--worker', nargs=3, help=argparse.SUPPRESS)
    args = p.parse_args()
    plan = verify(include_wheels=not bool(args.worker))
    if args.check:
        s = runtime()
        from bridge import SharedPolicy, IndependentPolicy
        for name, policy_type in [('shared', SharedPolicy), ('independent', IndependentPolicy), ('ppo', s.p.PilotPolicy)]:
            policy = policy_type(PKG / 'models' / name)
            if len(policy.models) != 1:
                raise ValueError('Unexpected model count')
            print('[READY]', name, 'members=1', flush=True)
        print('[READY] assets and dependencies verified; no objective evaluations', flush=True)
        return
    out = (args.out or PKG / ('computed_quick' if args.quick else 'computed_full')).resolve()
    if out == PKG or out in PKG.parents or out == PKG/'results' or PKG/'results' in out.parents:
        raise ValueError('Use a separate computed output directory')
    if args.worker:
        arm, instance, seed = args.worker
        if arm not in ARMS or int(seed) not in plan['settings']['screen_seeds']:
            raise ValueError('Unscheduled task')
        worker(plan, arm, instance, int(seed), out)
        return
    entries = plan['entries'][:1] if args.quick else plan['entries']
    seeds = plan['settings']['screen_seeds'][:1] if args.quick else plan['settings']['screen_seeds']
    out.mkdir(parents=True, exist_ok=True)
    scope = dict(package_plan_sha=sha(PKG/'assets/PLAN.json'), instances=[e['instance_id'] for e in entries], seeds=seeds)
    if (out/'SCOPE.json').exists() and read(out/'SCOPE.json') != scope:
        raise ValueError('Output belongs to a different run scope')
    write(out/'SCOPE.json', scope)
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [(arm, e['instance_id'], seed) for e in entries for seed in seeds for arm in ARMS]
        def launch(job):
            arm, name, seed = job
            dest = out/'runs'/name/('seed_'+str(seed))/arm
            if (dest/'PORTABLE_COMPLETE.json').exists():
                done = read(dest/'PORTABLE_COMPLETE.json')
                if done != dict(plan_sha=scope['package_plan_sha'], run_sha=sha(dest/'RUN_COMPLETE.json')):
                    raise ValueError('Completed result changed')
                return
            if dest.exists():
                dest.rename(dest.with_name(arm+'.interrupted-'+str(time.time_ns())))
            log = out/'logs'/(name+'_'+str(seed)+'_'+arm+'.log')
            log.parent.mkdir(exist_ok=True)
            with log.open('w') as f:
                subprocess.run([sys.executable, '-u', str(PKG/'run.py'), '--out', str(out),
                                '--worker', arm, name, str(seed)], stdout=f, stderr=subprocess.STDOUT, check=True)
        print('[SINGLE]', len(jobs), 'searches; workers=', args.workers, flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(launch, j) for j in jobs}
            finished = 0
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for f in done:
                    try:
                        f.result()
                    except Exception:
                        for remaining in pending:
                            remaining.cancel()
                        raise
                    finished += 1
                print('[SINGLE]', finished, '/', len(jobs), flush=True)
                write(out/'STATUS.json', dict(completed=False, done=finished, total=len(jobs)))
        report(plan, out, entries, seeds)
        print('[COMPLETE]', out/'COMPUTED_RESULTS.csv', flush=True)


if __name__ == '__main__':
    main()
