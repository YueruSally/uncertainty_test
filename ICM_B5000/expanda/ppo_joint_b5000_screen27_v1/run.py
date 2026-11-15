#!/usr/bin/env python3
"""Evaluate the fixed pilot u1 on the original 27 development pairs. No training."""
import argparse
import copy
import fcntl
import importlib.util
from pathlib import Path
import signal

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('screen27_audit', HERE.parent / 'ppo_joint_b5000_base_report_v1/run.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
p = audit.pilot
scheduler = p.scheduler
scheduler.HERE2 = HERE
read, write, freeze, digest, canonical = audit.read, audit.write, audit.freeze, audit.digest, audit.canonical
BUDGETS = audit.BUDGETS
NAMES = {'BASE': 'Base', 'SHARED_GREEDY': 'ICM 共享监督 Transformer', 'PPO_GREEDY': 'ICM PPO u1（最高分选择）'}
POOL_ARMS = tuple(p.hv.NAMES)[:-1]  # Previous u0 aliases add no points to the union.


def checked_plan(root):
    plan = read(root / 'PLAN.json')
    if canonical({k: v for k, v in plan.items() if k != 'sha256'}) != plan['sha256']:
        raise ValueError('Plan hash changed: ' + str(root))
    p.verify(plan['files'])
    return plan


def prepare(root, source):
    pilot_plan = checked_plan(source)
    status = read(source / 'STATUS.json'); selected = read(source / 'SELECTION.json')
    if not status.get('completed') or status['selected_updated_checkpoint'] != 1 or selected['best_updated_checkpoint'] != 1:
        raise ValueError('Expected the completed pilot with fixed best-updated u1')
    original = Path(pilot_plan['source']); old = checked_plan(original)
    if old['sha256'] != pilot_plan['source_plan_sha256'] or not read(original / 'STATUS.json').get('completed'):
        raise ValueError('Original B5000 panel changed or is incomplete')
    if old['normalization'] != pilot_plan['normalization']: raise ValueError('Normalization mismatch')
    settings = copy.deepcopy(old['settings']); entries = old['screen']; seeds = settings['screen_seeds']
    if (settings['population'], settings['budget'], settings['scenario_count'], settings['progress_denominator']) != (250, 5000, 100, 50000):
        raise ValueError('Expected pop250/B5000/CCP100 and frozen /50000 inputs')
    if tuple(settings['milestones']) != BUDGETS: raise ValueError('Snapshot schedule changed')
    if (sorted(e['configuration'] for e in entries) != ['S'+str(i) for i in range(9)]
            or any(e['development_role'] != 'screen' for e in entries)
            or len({e['instance_id'] for e in entries}) != 9 or list(seeds) != [993001, 993002, 993003]):
        raise ValueError('Expected the original S0-S8 development panel and three common seeds')
    model = source / 'checkpoints/update_001'; policy = p.PilotPolicy(model)
    if policy.bundle['normalization'] != old['normalization']: raise ValueError('Model normalization differs')
    if policy.bundle['settings']['progress_denominator'] != 50000: raise ValueError('Model progress differs')
    filemap = dict(pilot_plan['files'])
    for path in [source/'PLAN.json', source/'SELECTION.json', source/'STATUS.json', model/'BUNDLE.json',
                 original/'PLAN.json', *HERE.glob('*.py'), HERE/'run.sh', Path(audit.__file__)]:
        filemap[str(path.resolve())] = digest(path)
    filemap.update({str(model / name): sha for name, sha in policy.bundle['checkpoint_sha256'].items()})
    references = {'BASE': [], 'SHARED_GREEDY': []}; pool = []; scenarios = {}; model_hashes = {}
    for entry in entries:
        name = entry['instance_id']; info, _ = p.read_instance(original / 'instances' / (name+'.json'))
        if info != entry: raise ValueError('Original instance changed')
        freeze(root / 'instances' / (name+'.json'), info)
        filemap[str(root / 'instances' / (name+'.json'))] = digest(root / 'instances' / (name+'.json'))
        for seed in seeds:
            base = read(original / 'screen' / name / f'seed_{seed}/BASE/RUN_COMPLETE.json')
            for arm in POOL_ARMS:
                path = original / 'screen' / name / f'seed_{seed}' / arm / 'RUN_COMPLETE.json'; row = read(path)
                audit.validate_run(row, entry, seed, settings, base['scenario_digest'])
                if row['arm'] != arm or row['phase'] != 'screen' or row['stochastic'] or row['ppo_updates'] or row['plan_sha256'] != old['sha256']:
                    raise ValueError('Invalid original frozen run: ' + str(path))
                model_hashes.setdefault(arm, set()).add(row['model_bundle_sha256'])
                if arm == 'ICM_SHARED_SUPERVISED' and row['model_bundle_sha256'] != digest(Path(old['source'])/'shared_model/BUNDLE.json'):
                    raise ValueError('Shared control model changed')
                key = {'BASE':'BASE', 'ICM_SHARED_SUPERVISED':'SHARED_GREEDY'}.get(arm)
                if key: references[key].append(str(path))
                pool.append(str(path)); filemap[str(path)] = digest(path)
            scenarios[f'{name}/{seed}'] = base['scenario_digest']
    if any(len(v) != 1 for v in model_hashes.values()): raise ValueError('A source method changed model across runs')
    plan = dict(protocol='fixed-pilot-u1-original-27-pairs-v1', source=str(source), source_plan_sha256=pilot_plan['sha256'],
        original=str(original), original_plan_sha256=old['sha256'], model=str(model), model_bundle_sha256=digest(model/'BUNDLE.json'),
        selected_update=1, original_selection=selected, settings=settings, normalization=old['normalization'],
        entries=entries, references=references, union_pool=pool, scenarios=scenarios, files=filemap,
        maximum_new_searches=27, maximum_new_objective_evaluations=135000,
        qualification='Fixed u1, deterministic original pilot selector. Previously inspected S0-S8 development panel; not independent final testing. No retraining or checkpoint reselection.',
        union_definition='Original ten-method fronts plus u1, all three seeds and all three budgets per instance; reference 1.1; no division by 1.331; report only Base/shared/PPO.')
    plan['sha256'] = canonical(plan); freeze(root / 'PLAN.json', plan)
    return plan


def task(plan, entry, seed):
    return p.make_task(plan, entry, seed, Path(plan['model']), 'screen', False, False, 1)


def validate_task(root, plan, t, folder):
    entry = next((e for e in plan['entries'] if e['instance_id'] == t['instance']), None)
    if entry is None or entry['development_role'] != 'screen' or t['seed'] not in plan['settings']['screen_seeds']:
        raise ValueError('Instance/seed outside frozen screen panel')
    expected = root / 'runs' / entry['instance_id'] / f'seed_{t["seed"]}' / 'PPO_U001'
    if t != task(plan, entry, t['seed']) or expected.resolve() != folder.resolve():
        raise ValueError('Only the fixed u1 greedy evaluation is permitted')
    if t['bundle_sha256'] != plan['model_bundle_sha256']: raise ValueError('Selected u1 changed')
    return entry


def validate_ppo(row, plan):
    entry = next(e for e in plan['entries'] if e['instance_id'] == row['instance'])
    audit.validate_run(row, entry, row['seed'], plan['settings'], plan['scenarios'][f'{row["instance"]}/{row["seed"]}'])
    expected = task(plan, entry, row['seed'])
    if (row['plan_sha256'] != plan['sha256'] or row['task_sha256'] != canonical(expected)
            or row['stochastic'] or row['trajectory_sha256'] or row['model_bundle_sha256'] != plan['model_bundle_sha256']):
        raise ValueError('PPO output provenance changed')


def worker(root, plan, t, folder):
    validate_task(root, plan, t, folder)
    # Reuse the exact v2 policy/search worker. Only its allowed task schedule changes.
    original_validator = p.validate_task
    p.validate_task = validate_task
    try: p.run_worker(root, plan, t, folder)
    finally: p.validate_task = original_validator
    validate_ppo(read(folder / 'RUN_COMPLETE.json'), plan)


def groups_for(rows, plan):
    for row in rows: validate_ppo(row, plan)
    keys = {(r['instance'], r['seed']) for r in rows}
    if len(keys) != len(rows): raise ValueError('Duplicated PPO pair')
    groups = {'PPO_GREEDY': rows}
    for method, paths in plan['references'].items():
        groups[method] = [r for r in (read(Path(path)) for path in paths) if (r['instance'], r['seed']) in keys]
    return groups


def report(groups, destination, plan, partial=False, union=False):
    main = []; direct = []; details = {}; per_run = []; per_config = []
    for budget in BUDGETS:
        at = {m: [dict(r, metric=next(s['hv'] for s in r['budget_fronts'] if s['evaluation']==budget)) for r in rows] for m, rows in groups.items()}
        for method, reference in [('BASE','BASE'), ('SHARED_GREEDY','BASE'), ('PPO_GREEDY','BASE'), ('PPO_GREEDY','SHARED_GREEDY')]:
            v = p.paired(at[method], at[reference], field='metric', draws=1 if partial else 3000)
            lo, hi = (None, None) if partial else p.hv.interval(v['per_run'], 'percent', 3000)
            if partial: v['absolute_ci_low'] = v['absolute_ci_high'] = None
            v.update(percent_ci_low=lo, percent_ci_high=hi); details[f'{method}/{reference}/{budget}'] = v
            row = {'评价次数':budget, '方法':NAMES[method], '参照':NAMES[reference], '运行对数':v['pairs'],
                '平均HV':v['mean_hv'], '平均配对提升(%)':v['mean_paired_percent'], '提升95%下限(%)':lo, '提升95%上限(%)':hi,
                '平均HV绝对差':v['mean_absolute_difference'], '绝对差95%下限':v['absolute_ci_low'], '绝对差95%上限':v['absolute_ci_high'],
                '胜':v['wins'], '平':v['ties'], '负':v['losses'],
                '平均B5000秒':p.equal_mean(at[method], 'optimisation_seconds') if budget==5000 else None}
            (main if reference=='BASE' else direct).append(row)
            per_run.extend(dict(method=method, reference_method=reference, evaluation=budget, **r) for r in v['per_run'])
            for cfg in sorted({r['configuration'] for r in v['per_run']}):
                cfg_rows = [r for r in v['per_run'] if r['configuration']==cfg]
                per_config.append(dict(method=method, reference=reference, evaluation=budget, configuration=cfg,
                    pairs=len(cfg_rows), mean_absolute=p.equal_mean(cfg_rows, 'absolute'),
                    mean_percent=p.hv.equal_mean(cfg_rows, 'percent')))
    destination.mkdir(parents=True, exist_ok=True)
    p.csv_write(destination/'00_共享监督与PPO_相对Base.csv', main)
    p.csv_write(destination/'01_PPO相对共享监督.csv', direct)
    p.csv_write(destination/'PER_RUN.csv', per_run); p.csv_write(destination/'PER_CONFIGURATION.csv', per_config)
    write(destination/'QUICK_RESULT.json', dict(completed=not partial, preview_only=partial, comparisons=details,
        selected_update=1, model_bundle_sha256=plan['model_bundle_sha256'], qualification=plan['qualification'],
        normalization=plan['union_definition'] if union else 'unchanged frozen training normalization',
        interval_scope='Paired bootstrap within fixed configurations. One instance/configuration: intervals reflect seeds on these instances, not the configuration population.'))
    print(f'[PPO-27] {"PREVIEW" if partial else "REPORT"} {destination}', flush=True)
    for row in direct: print('[PPO-27] ' + p.json.dumps(row, ensure_ascii=False), flush=True)


def execute(a):
    root = a.out.resolve(); source = a.source.resolve()
    if root==source or root in source.parents or source in root.parents: raise ValueError('Use a separate output directory')
    root.mkdir(parents=True, exist_ok=True); p.torch.set_num_threads(1)
    with (root/'.pipeline.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('PPO 27-pair evaluation already running')
        def stop(signum, frame): raise scheduler.Paused(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        try:
            plan = prepare(root, source)
            jobs = [(task(plan,e,s), root/'runs'/e['instance_id']/f'seed_{s}'/'PPO_U001')
                    for e in sorted(plan['entries'], key=lambda e:e['configuration']) for s in plan['settings']['screen_seeds']]
            print('[PPO-27] 27 frozen-u1 searches; reuse 54 matched Base/shared controls; no training; first 2 pairs preview', flush=True)
            early = scheduler.execute_jobs(root, plan, jobs[:2], a.workers, 'screen27/preview_2_of_27')
            report(groups_for(early, plan), root/'preview', plan, partial=True)
            rows = scheduler.execute_jobs(root, plan, jobs, a.workers, 'screen27/all_27')
            groups = groups_for(rows, plan)
            if len(rows) != 27: raise ValueError('Incomplete 27-pair panel')
            report(groups, root/'results', plan)
            # Keep the original full method pool in the bounds for continuity with the old table.
            pool = {'ORIGINAL_TEN_METHODS': [read(Path(path)) for path in plan['union_pool']], 'PPO_GREEDY': rows}
            normalized, bounds = audit.union_normalize(pool)
            old_rows = normalized['ORIGINAL_TEN_METHODS']
            union = {'BASE':[r for r in old_rows if r['arm']=='BASE'],
                     'SHARED_GREEDY':[r for r in old_rows if r['arm']=='ICM_SHARED_SUPERVISED'],
                     'PPO_GREEDY':normalized['PPO_GREEDY']}
            for b in bounds.values(): b['methods'] = list(POOL_ARMS)+['PPO_PILOT_V2_U001']
            report(union, root/'results_union', plan, union=True)
            write(root/'results_union/INSTANCE_BOUNDS.json', bounds)
            write(root/'STATUS.json', dict(completed=True, stage='all', done=27, total=27, selected_update=1,
                reused_base_runs=27, reused_shared_runs=27, physical_ppo_runs=27, new_ppo_updates=0,
                completed_ppo_evaluations=sum(r['real_objective_evaluation_count'] for r in rows),
                hidden_candidate_objective_evaluations=0, final_test_opened=False))
            print('[PPO-27] COMPLETE', flush=True)
        except scheduler.Paused as e:
            write(root/'STATUS.json', dict(completed=False, paused=True, reason=str(e), resume='same command; completed runs reused'))
            print('[PPO-27] PAUSED '+str(e), flush=True)
        except Exception as e:
            write(root/'STATUS.json', dict(completed=False, failed=True, error=str(e))); raise


def main():
    diag = HERE.parent/'transformer_ppo_runs/diagnostics'; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=diag/'ccp_joint_ppo_b5000_pilot_v2')
    parser.add_argument('--out', type=Path, default=diag/'ccp_joint_ppo_b5000_screen27_v1')
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--worker', action='store_true')
    parser.add_argument('--task', type=Path); parser.add_argument('--run-out', type=Path); a = parser.parse_args()
    if a.workers<1: parser.error('workers must be positive')
    if a.worker: worker(a.out, checked_plan(a.out), read(a.task), a.run_out)
    else: execute(a)


if __name__=='__main__': main()
