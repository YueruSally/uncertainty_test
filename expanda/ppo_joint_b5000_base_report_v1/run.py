#!/usr/bin/env python3
"""Add matched BASE controls to the completed PPO pilot; no training or reselection."""
from pathlib import Path
import argparse
import copy
import fcntl
import importlib.util
import json
import os
import signal
import sys

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'ppo_joint_b5000_pilot_v2'
sys.path.insert(0, str(PILOT))
spec = importlib.util.spec_from_file_location('base_report_pilot', PILOT / 'run.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
from search_b5000 import run_search

read = pilot.read_json
write = pilot.write_json
digest = pilot.digest
freeze = pilot.freeze
canonical = pilot.canonical_digest
scheduler = pilot.scheduler
scheduler.HERE2 = HERE
BUDGETS = (400, 1000, 5000)
NAMES = {'BASE': 'Base', 'SHARED_GREEDY': 'ICM 共享监督_最高分',
         'PPO_GREEDY': 'ICM PPO u{update}_最高分',
         'SHARED_SAMPLED': 'ICM 共享监督_概率采样',
         'PPO_SAMPLED': 'ICM PPO u{update}_概率采样'}


def validate_run(row, entry, seed, settings, scenario=None):
    if (not row['completed'] or row['instance'] != entry['instance_id']
            or row['instance_digest'] != entry['instance_digest'] or row['seed'] != seed
            or row['configuration'] != entry['configuration']
            or row['evaluation_budget'] != settings['budget']
            or not settings['budget'] - 1 <= row['real_objective_evaluation_count'] <= settings['budget']
            or row['hidden_candidate_objective_evaluations']):
        raise ValueError('Unmatched/incomplete run: ' + str((entry['instance_id'], seed)))
    if scenario is not None and row['scenario_digest'] != scenario:
        raise ValueError('Paired scenarios differ')
    snaps = row['budget_fronts']
    if len(snaps) != len(settings['milestones']) or {p['evaluation'] for p in snaps} != set(settings['milestones']):
        raise ValueError('Missing or duplicate snapshots')
    final = next(p for p in snaps if p['evaluation'] == settings['budget'])
    if final['front_raw'] != row['final_front_raw'] or final['hv'] != row['final_hypervolume']:
        raise ValueError('Final front/HV mismatch')


def prepare(root, source):
    source_plan = read(source / 'PLAN.json')
    if canonical({k: v for k, v in source_plan.items() if k != 'sha256'}) != source_plan['sha256']:
        raise ValueError('Pilot plan changed')
    pilot.verify(source_plan['files'])
    status = read(source / 'STATUS.json')
    if not status.get('completed'): raise ValueError('Pilot must be complete')
    choice = read(source / 'SELECTION.json')
    update = choice['best_updated_checkpoint']
    if update not in (1, 2, 3) or status['selected_updated_checkpoint'] != update:
        raise ValueError('Unexpected selected updated model')
    settings = copy.deepcopy(source_plan['settings'])
    if (settings['population'], settings['budget'], settings['scenario_count'], settings['progress_denominator']) != (250, 5000, 100, 50000):
        raise ValueError('Expected unchanged pop250/B5000/CCP100 inputs')
    entries = source_plan['confirmation']
    seeds = settings['ppo']['validation_seeds']
    if sorted(e['configuration'] for e in entries) != ['S0', 'S4', 'S8'] or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Expected three confirmation instances and three seeds')
    if any(e['development_role'] != 'validation' for e in entries): raise ValueError('Wrong instance role')
    files = dict(source_plan['files'])
    for p in [source / 'PLAN.json', source / 'SELECTION.json', source / 'STATUS.json', *HERE.glob('*.py')]:
        files[str(p.resolve())] = digest(p)
    sources = {key: [] for key in NAMES if key != 'BASE'}
    baseline_map = {}
    for name in source_plan['validation_baseline_files']:
        row = read(Path(name))
        baseline_map[row['instance'], row['seed']] = Path(name)
    scenarios = {}
    for entry in entries:
        instance = entry['instance_id']
        info, _ = pilot.read_instance(source / 'instances' / (instance + '.json'))
        if info != entry: raise ValueError('Source instance changed')
        freeze(root / 'instances' / (instance + '.json'), entry)
        files[str(root / 'instances' / (instance + '.json'))] = digest(root / 'instances' / (instance + '.json'))
        for seed in seeds:
            paths = {'SHARED_GREEDY': baseline_map[instance, seed],
                'PPO_GREEDY': source / f'runs/confirmation/u{update:03d}/greedy' / instance / f'seed_{seed}/RUN_COMPLETE.json',
                'PPO_SAMPLED': source / f'runs/confirmation/u{update:03d}/sampled' / instance / f'seed_{seed}/RUN_COMPLETE.json',
                'SHARED_SAMPLED': source / 'runs/confirmation/u000/sampled' / instance / f'seed_{seed}/RUN_COMPLETE.json'}
            shared = read(paths['SHARED_GREEDY'])
            for method, path in paths.items():
                row = read(path)
                validate_run(row, entry, seed, settings, shared['scenario_digest'])
                if method != 'SHARED_GREEDY':
                    number = 0 if method == 'SHARED_SAMPLED' else update
                    model = source / 'checkpoints' / f'update_{number:03d}' / 'BUNDLE.json'
                    if row['plan_sha256'] != source_plan['sha256'] or row['model_bundle_sha256'] != digest(model):
                        raise ValueError('Confirmation model/provenance changed')
                    files[str(model)] = digest(model)
                if bool(row['stochastic']) != method.endswith('SAMPLED'): raise ValueError('Deployment differs')
                sources[method].append(str(path)); files[str(path)] = digest(path)
            scenarios[f'{instance}/{seed}'] = shared['scenario_digest']

    # Only inspect exact instance/seed paths in the original frozen B5000 study.
    # Its screen instances normally differ; never substitute an overall BASE mean.
    original = Path(source_plan['source']); original_plan = read(original / 'PLAN.json')
    if original_plan['sha256'] != source_plan['source_plan_sha256'] or original_plan['normalization'] != source_plan['normalization']:
        raise ValueError('Original comparison provenance differs')
    reused = {}
    for entry in entries:
        for seed in seeds:
            key = f'{entry["instance_id"]}/{seed}'
            path = original / 'screen' / entry['instance_id'] / f'seed_{seed}/BASE/RUN_COMPLETE.json'
            if not path.exists(): continue
            row = read(path); validate_run(row, entry, seed, settings, scenarios[key])
            if row['arm'] != 'BASE' or row['stochastic'] or row['plan_sha256'] != original_plan['sha256']:
                raise ValueError('Invalid reusable Base')
            reused[key] = str(path); files[str(path)] = digest(path)
    plan = dict(source=str(source), source_plan_sha256=source_plan['sha256'], settings=settings,
        normalization=source_plan['normalization'], entries=entries, sources=sources,
        scenarios=scenarios, reused_base=reused, selected_update=update, files=files,
        protocol='paired-base-controls-for-fixed-pilot-v1', reference_method='BASE',
        qualification='Development confirmation only; original checkpoint/deployment choices unchanged. No training or final test.')
    plan['sha256'] = canonical(plan); freeze(root / 'PLAN.json', plan)
    return plan


def base_task(plan, entry, seed):
    return scheduler.task(plan, entry['instance_id'], seed, 'BASE', 'validation')


def worker(root, plan, task, folder):
    pilot.verify(plan['files'])
    entry = next((e for e in plan['entries'] if e['instance_id'] == task['instance']), None)
    if entry is None or task['seed'] not in plan['settings']['ppo']['validation_seeds']:
        raise ValueError('Unscheduled instance/seed')
    expected = root / 'runs' / entry['instance_id'] / f'seed_{task["seed"]}' / 'BASE'
    if task != base_task(plan, entry, task['seed']) or expected.resolve() != folder.resolve():
        raise ValueError('Worker may only run the fixed BASE control')
    row = run_search(root / 'instances' / (entry['instance_id'] + '.json'), plan['settings'], task, folder, plan['sha256'])
    validate_run(row, entry, task['seed'], plan['settings'], plan['scenarios'][f'{entry["instance_id"]}/{task["seed"]}'])
    if any(row['selection'][c]['calls'] for c in 'ICM'): raise ValueError('Base used a learned/rule hook')


def union_normalize(groups):
    result = copy.deepcopy(groups); bounds = {}
    for instance in sorted({r['instance'] for records in result.values() for r in records}):
        records = [r for runs in result.values() for r in runs if r['instance'] == instance]
        points = [p for r in records for s in r['budget_fronts'] for p in s['front_raw']]
        if points:
            data = pilot.np.asarray(points, dtype=float)
            if data.ndim != 2 or data.shape[1] != 3 or not pilot.np.isfinite(data).all(): raise ValueError('Invalid objectives')
            low, high = data.min(0), data.max(0)
        else: low = high = pilot.np.zeros(3)
        span = pilot.np.where(high > low, high - low, 1.)
        bounds[instance] = dict(minimum=low.tolist(), maximum=high.tolist(), reference=[1.1]*3,
            budgets=list(BUDGETS), methods=list(groups), seeds=sorted({r['seed'] for r in records}), includes_base=True)
        for row in records:
            for snap in row['budget_fronts']:
                points = [(pilot.np.asarray(p) - low) / span for p in snap['front_raw']]
                if not all(pilot.np.all((p >= 0) & (p <= 1)) for p in points): raise ValueError('Out of union bounds')
                snap['hv'] = pilot.hv.hv3(points)
            row['final_hypervolume'] = next(s['hv'] for s in row['budget_fronts'] if s['evaluation'] == 5000)
    return result, bounds


def report(groups, out, plan, union=False):
    rows = []; details = {}; per_run = []
    for budget in BUDGETS:
        at = {m: [dict(r, metric=next(p['hv'] for p in r['budget_fronts'] if p['evaluation'] == budget)) for r in records]
              for m, records in groups.items()}
        for method in NAMES:
            own = at[method]; comparison = pilot.paired(own, at['BASE'], field='metric')
            lo, hi = pilot.hv.interval(comparison['per_run'], 'percent', 3000)
            name = NAMES[method].format(update=plan['selected_update'])
            rows.append({'评价次数': budget, '方法': name, '参照': 'Base', '运行对数': comparison['pairs'],
                '平均HV': comparison['mean_hv'], '最大HV': max(r['metric'] for r in own),
                '相对Base平均配对提升(%)': comparison['mean_paired_percent'],
                '提升95%下限(%)': lo, '提升95%上限(%)': hi,
                '平均HV绝对差': comparison['mean_absolute_difference'],
                '绝对差95%下限': comparison['absolute_ci_low'], '绝对差95%上限': comparison['absolute_ci_high'],
                '胜': comparison['wins'], '平': comparison['ties'], '负': comparison['losses'],
                '平均B5000秒': pilot.equal_mean(own, 'optimisation_seconds') if budget == 5000 else None})
            details[f'{method}/BASE/{budget}'] = dict(comparison, percent_ci_low=lo, percent_ci_high=hi)
            for r in comparison['per_run']:
                per_run.append(dict(method=method, evaluation=budget, **r))
        details[f'PPO_GREEDY/SHARED_GREEDY/{budget}'] = pilot.paired(at['PPO_GREEDY'], at['SHARED_GREEDY'], field='metric')
        details[f'PPO_SAMPLED/SHARED_SAMPLED/{budget}'] = pilot.paired(at['PPO_SAMPLED'], at['SHARED_SAMPLED'], field='metric')
    out.mkdir(parents=True, exist_ok=True)
    pilot.csv_write(out / '00_相对Base结果.csv', rows); pilot.csv_write(out / 'PER_RUN.csv', per_run)
    write(out / 'QUICK_RESULT.json', dict(completed=True, reference_method='BASE', selected_update=plan['selected_update'],
        normalization='instance union including BASE; reference (1.1,1.1,1.1); not divided by 1.331' if union else 'unchanged frozen training normalization',
        comparisons=details, qualification=plan['qualification'],
        interval_scope='paired bootstrap within fixed configurations; one confirmation instance per configuration; seed variability only'))
    print('[PPO-BASE] ' + str(out / '00_相对Base结果.csv'), flush=True)
    for row in rows:
        if 'PPO u' in row['方法'] and '最高分' in row['方法']:
            print('[PPO-BASE] ' + json.dumps(row, ensure_ascii=False), flush=True)


def execute(args):
    root = args.out.resolve(); source = args.source.resolve()
    if root == source or source in root.parents or root in source.parents: raise ValueError('Separate output required')
    root.mkdir(parents=True, exist_ok=True); pilot.torch.set_num_threads(1)
    with (root / '.pipeline.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Base supplement already running')
        def stop(signum, frame): raise scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            plan = prepare(root, source); jobs = []
            for entry in plan['entries']:
                for seed in plan['settings']['ppo']['validation_seeds']:
                    if f'{entry["instance_id"]}/{seed}' in plan['reused_base']: continue
                    jobs.append((base_task(plan, entry, seed), root / 'runs' / entry['instance_id'] / f'seed_{seed}' / 'BASE'))
            print(f'[PPO-BASE] {len(jobs)} BASE controls; {len(plan["reused_base"])} exact reusable controls; no PPO training or reselection', flush=True)
            physical = scheduler.execute_jobs(root, plan, jobs, args.workers, 'paired_BASE')
            groups = {'BASE': physical + [read(Path(p)) for p in plan['reused_base'].values()]}
            groups.update({m: [read(Path(p)) for p in paths] for m, paths in plan['sources'].items()})
            for row in groups['BASE']:
                entry = next(e for e in plan['entries'] if e['instance_id'] == row['instance'])
                validate_run(row, entry, row['seed'], plan['settings'], plan['scenarios'][f'{row["instance"]}/{row["seed"]}'])
            report(groups, root / 'results', plan)
            normalized, bounds = union_normalize(groups)
            report(normalized, root / 'results_union', plan, union=True)
            write(root / 'results_union/INSTANCE_BOUNDS.json', bounds)
            write(root / 'STATUS.json', dict(completed=True, stage='all', reference_method='BASE', selected_update=plan['selected_update'],
                base_physical_runs=len(physical), base_reused_from_source=len(plan['reused_base']),
                source_method_runs_reused=sum(len(r) for m, r in groups.items() if m != 'BASE'),
                new_ppo_updates=0, completed_base_objective_evaluations=sum(r['real_objective_evaluation_count'] for r in physical),
                final_test_opened=False))
            print('[PPO-BASE] COMPLETE', flush=True)
        except scheduler.Paused as e:
            write(root / 'STATUS.json', dict(completed=False, paused=True, reason=str(e), resume='repeat same command'))
        except Exception as e:
            write(root / 'STATUS.json', dict(completed=False, failed=True, error=str(e))); raise


def main():
    diag = HERE.parent / 'transformer_ppo_runs/diagnostics'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=diag / 'ccp_joint_ppo_b5000_pilot_v2')
    parser.add_argument('--out', type=Path, default=diag / 'ccp_joint_ppo_b5000_pilot_v2_vs_base_v1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker', action='store_true'); parser.add_argument('--task', type=Path); parser.add_argument('--run-out', type=Path)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    if args.worker:
        plan = read(args.out / 'PLAN.json')
        if canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']: raise ValueError('Plan changed')
        worker(args.out, plan, read(args.task), args.run_out)
    else: execute(args)


if __name__ == '__main__': main()
