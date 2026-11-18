"""Read-only B5000 postprocessing. No search, policy import, or reward changes."""
import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

BUDGETS = (400, 1000, 5000)
REF = (1.1, 1.1, 1.1)
VOLUME = math.prod(REF)
NAMES = dict(BASE='Base', I_RULE='I 启发式', I_LEARN='I Transformer',
             C_RULE='C 启发式', C_LEARN='C Transformer',
             M_RULE='M 启发式', M_LEARN='M Transformer',
             ICM_HEURISTIC='ICM 启发式', ICM_INDEPENDENT='ICM 独立 Transformer',
             ICM_SHARED_SUPERVISED='ICM 共享监督 Transformer',
             ICM_TERMINAL_PPO='ICM 终局联合 PPO')
ARMS = tuple(NAMES)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hv3(points):
    """Exact union of minimization boxes [point, REF], by x/y sweeps.

    Same slab geometry as transformer_ppo/archive.py; accepts dominated points.
    Input validation is performed before this function. No external HV package.
    """
    points = sorted(set(tuple(p) for p in points))
    volume, active, i = 0.0, [], 0
    while i < len(points):
        x = points[i][0]
        while i < len(points) and points[i][0] == x:
            active.append(points[i][1:])
            i += 1
        next_x = points[i][0] if i < len(points) else REF[0]
        yz = sorted(active)
        best_z, area, j = REF[2], 0.0, 0
        while j < len(yz):
            y = yz[j][0]
            while j < len(yz) and yz[j][0] == y:
                best_z = min(best_z, yz[j][1])
                j += 1
            next_y = yz[j][0] if j < len(yz) else REF[1]
            area += (next_y-y) * (REF[2]-best_z)
        volume += (next_x-x) * area
    return float(volume)


def validate_panel(physical, plan, report):
    require(report.get('completed') and not report.get('preview_only'), 'Only a completed report is allowed')
    settings = plan['settings']
    require(settings['budget'] == 5000 and settings['population'] == 250
            and tuple(settings['milestones']) == BUDGETS, 'Expected pop250/B5000 and 400/1000/5000 snapshots')
    entries = {r['instance_id']: r for r in plan['screen']}
    require(len(entries) == len(plan['screen']), 'Duplicate planned instance')
    seeds = settings['screen_seeds']
    require(len(set(seeds)) == len(seeds), 'Duplicate planned seed')
    indexed = {}
    for r in physical:
        key = (r['instance'], r['seed'], r['arm'])
        require(key not in indexed, 'Duplicate run: '+str(key))
        require(r['instance'] in entries and r['seed'] in seeds and r['arm'] in ARMS, 'Unplanned run')
        require(r['completed'] and r['plan_sha256'] == plan['sha256'] and r['phase'] == 'screen'
                and not r['stochastic'] and not r['hidden_candidate_objective_evaluations']
                and not r['ppo_updates'] and r['evaluation_budget'] == 5000
                and 4999 <= r['real_objective_evaluation_count'] <= 5000, 'Invalid completed frozen run: '+str(key))
        entry = entries[r['instance']]
        require(r['configuration'] == entry['configuration'] and r['instance_digest'] == entry['instance_digest'],
                'Planned instance differs: '+str(key))
        require(r['scenario_digest'] and math.isfinite(r['optimisation_seconds']) and r['optimisation_seconds'] >= 0,
                'Missing scenario or invalid runtime')
        snaps = r['budget_fronts']
        require(len(snaps) == 3 and {p['evaluation'] for p in snaps} == set(BUDGETS), 'Missing/duplicate budget snapshot')
        for p in snaps:
            actual = p['actual_evaluations']
            require(actual == (r['real_objective_evaluation_count'] if p['evaluation'] == 5000 else p['evaluation']),
                    'Wrong snapshot evaluation count')
            require(isinstance(p['front_raw'], list), 'Missing raw feasible front')
            for point in p['front_raw']:
                require(len(point) == 3 and all(math.isfinite(float(x)) for x in point), 'Nonfinite/non-3D objective')
        final = next(p for p in snaps if p['evaluation'] == 5000)
        require(final['front_raw'] == r['final_front_raw'], 'Final snapshot differs from final front')
        indexed[key] = dict(r, result_source='physical')
    reuse = report.get('ppo_alias')
    if reuse:
        require(reuse['selected_update'] == 0 and report['selected_ppo_update'] == 0
                and reuse['alias_of'] == 'ICM_SHARED_SUPERVISED', 'Invalid PPO alias')
        checks = {(r['instance'], r['seed']) for r in physical if r['arm'] == 'ICM_TERMINAL_PPO'}
        claimed = {(r['instance'], r['seed']) for r in reuse['checked_keys']}
        require(checks == claimed and len(checks) == 2 and reuse['actual_reproduction_runs'] == 2
                and reuse['reused_rows'] == len(entries)*len(seeds)-2, 'Expected two physical u0 reproduction checks')
        for instance, seed in checks:
            a = indexed[instance, seed, 'ICM_SHARED_SUPERVISED']
            b = indexed[instance, seed, 'ICM_TERMINAL_PPO']
            for field in ('final_front_raw', 'final_hypervolume', 'evaluation_order_digest',
                          'real_objective_evaluation_count', 'instance_digest', 'scenario_digest', 'selection'):
                require(a[field] == b[field], 'u0 replay mismatch: '+field)
            for n in BUDGETS:
                pa = next(p for p in a['budget_fronts'] if p['evaluation'] == n)
                pb = next(p for p in b['budget_fronts'] if p['evaluation'] == n)
                require(pa['front_raw'] == pb['front_raw'], 'u0 snapshot mismatch')
        for r in list(indexed.values()):
            if r['arm'] == 'ICM_SHARED_SUPERVISED' and (r['instance'], r['seed']) not in checks:
                indexed[r['instance'], r['seed'], 'ICM_TERMINAL_PPO'] = dict(
                    r, arm='ICM_TERMINAL_PPO', result_source='u0_alias_of_shared', optimisation_seconds=None)
    methods = [m for m in ARMS if any(k[2] == m for k in indexed)]
    require(methods in [list(ARMS[:-1]), list(ARMS)], 'Missing method block')
    expected = {(i, s, m) for i in entries for s in seeds for m in methods}
    require(set(indexed) == expected, 'Incomplete instance/seed/method panel')
    for i in entries:
        for s in seeds:
            require(len({indexed[i, s, m]['scenario_digest'] for m in methods}) == 1, 'Paired scenarios differ')
    # Frozen models must not change between instances or seeds of an arm.
    for m in methods:
        if m == 'ICM_TERMINAL_PPO' and reuse:
            continue  # the physical u0 bundle has a distinct wrapper hash
        require(len({r.get('model_bundle_sha256') for r in indexed.values() if r['arm'] == m}) == 1,
                'Model bundle changed within method: '+m)
    return [indexed[k] for k in sorted(indexed)], methods


def renormalize(rows):
    """Bounds use ALL saved feasible fronts, not the globally filtered union."""
    bounds, per_run = {}, []
    for instance in sorted({r['instance'] for r in rows}):
        group = [r for r in rows if r['instance'] == instance]
        points = [p for r in group for snap in r['budget_fronts'] for p in snap['front_raw']]
        if points:
            data = np.asarray(points, dtype=float)
            lo, hi = data.min(axis=0), data.max(axis=0)
        else:
            lo, hi = np.zeros(3), np.zeros(3)
        width = np.where(hi > lo, hi-lo, 1.0)
        bounds[instance] = dict(minimum=lo.tolist(), maximum=hi.tolist(), effective_width=width.tolist(),
            constant_objectives=np.flatnonzero(hi == lo).tolist(), empty_union=not bool(points),
            reference=list(REF), reference_box_volume=VOLUME, budgets=list(BUDGETS),
            methods=sorted({r['arm'] for r in group}), seeds=sorted({r['seed'] for r in group}))
        for r in group:
            for snap in sorted(r['budget_fronts'], key=lambda p: p['evaluation']):
                normalized = [(np.asarray(p, dtype=float)-lo)/width for p in snap['front_raw']]
                require(all(np.all((p >= 0) & (p <= 1)) for p in normalized), 'Normalized coordinate outside [0,1]')
                raw_hv = hv3(normalized)
                require(-1e-12 <= raw_hv <= VOLUME+1e-12, 'HV outside reference box')
                per_run.append(dict(instance=instance, configuration=r['configuration'], seed=r['seed'], arm=r['arm'],
                    evaluation=snap['evaluation'], actual_evaluations=snap['actual_evaluations'],
                    hv_union=raw_hv, hv_01=raw_hv/VOLUME, old_hv_frozen_training=snap['hv'],
                    controller_seconds=snap['controller_seconds'] if r['result_source'] == 'physical' else None,
                    full_run_seconds=r['optimisation_seconds'] if snap['evaluation'] == 5000 else None,
                    result_source=r['result_source'], instance_digest=r['instance_digest'], scenario_digest=r['scenario_digest']))
    return per_run, bounds


def equal_mean(rows, field):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r[field] is None:
            return None
        groups[r['configuration']][r['instance']].append(r[field])
    return float(np.mean([np.mean([np.mean(v) for v in inst.values()]) for inst in groups.values()]))


def interval(rows, field, draws):
    if any(r[field] is None for r in rows):
        return None, None
    rng = np.random.default_rng(953001)
    samples = []
    for c in sorted({r['configuration'] for r in rows}):
        instances = sorted({r['instance'] for r in rows if r['configuration'] == c})
        x = np.asarray([[r[field] for r in sorted(rows, key=lambda r: r['seed'])
                         if r['configuration'] == c and r['instance'] == i] for i in instances])
        ids = rng.integers(len(x), size=(draws, len(x)))
        seeds = rng.integers(x.shape[1], size=(draws, len(x), x.shape[1]))
        samples.append(x[ids[:, :, None], seeds].mean(axis=(1, 2)))
    return tuple(float(x) for x in np.quantile(np.mean(samples, axis=0), [.025, .975]))


def paired(rows, method, reference, draws):
    baseline = {(r['instance'], r['seed']): r for r in rows if r['arm'] == reference}
    pairs = []
    for r in rows:
        if r['arm'] != method:
            continue
        b = baseline[r['instance'], r['seed']]
        d = r['hv_01']-b['hv_01']
        pairs.append(dict(r, absolute=d, percent=100*d/b['hv_01'] if b['hv_01'] > 0 else None))
    p_lo, p_hi = interval(pairs, 'percent', draws)
    a_lo, a_hi = interval(pairs, 'absolute', draws)
    tolerance = 1e-12/VOLUME
    valid = [r for r in pairs if r['percent'] is not None]
    best = max(valid, key=lambda r: r['percent']) if valid else None
    return dict(method=method, reference=reference, pairs=len(pairs), mean_percent=equal_mean(pairs, 'percent'),
        ci_low=p_lo, ci_high=p_hi, mean_absolute_hv01=equal_mean(pairs, 'absolute'), absolute_ci_low=a_lo, absolute_ci_high=a_hi,
        wins=sum(r['absolute'] > tolerance for r in pairs), ties=sum(abs(r['absolute']) <= tolerance for r in pairs),
        losses=sum(r['absolute'] < -tolerance for r in pairs), zero_reference_pairs=len(pairs)-len(valid),
        max_percent=best['percent'] if best else None, max_percent_instance=best['instance'] if best else None,
        max_percent_seed=best['seed'] if best else None)


def tables(per_run, methods, alias, draws):
    summary, stats = [], []
    for n in BUDGETS:
        sample = [r for r in per_run if r['evaluation'] == n]
        for m in methods:
            own = [r for r in sample if r['arm'] == m]
            ref = m[0]+'_RULE' if m.startswith(('I_', 'C_', 'M_')) else ('BASE' if m == 'BASE' else 'ICM_HEURISTIC')
            refs = list(dict.fromkeys(['BASE', ref, 'ICM_INDEPENDENT', 'ICM_SHARED_SUPERVISED']))
            comparisons = {r: paired(sample, m, r, draws) for r in refs}
            stats.extend(dict(evaluation=n, **v) for v in comparisons.values())
            b, rule = comparisons['BASE'], comparisons[ref]
            best = max(own, key=lambda r: r['hv_01'])
            is_alias = bool(alias and m == 'ICM_TERMINAL_PPO')
            summary.append({'评价次数': n, '方法': NAMES[m], '平均HV_0至1': equal_mean(own, 'hv_01'),
                '单次最大HV_0至1': best['hv_01'], '平均配对提升Base(%)': b['mean_percent'],
                '单次最大配对提升Base(%)': b['max_percent'], 'Base提升95%下限(%)': b['ci_low'], 'Base提升95%上限(%)': b['ci_high'],
                '规则参照': NAMES[ref], '平均配对提升规则(%)': rule['mean_percent'],
                '规则提升95%下限(%)': rule['ci_low'], '规则提升95%上限(%)': rule['ci_high'],
                '胜Base': b['wins'], '平Base': b['ties'], '负Base': b['losses'], 'Base为零对数': b['zero_reference_pairs'],
                '平均原始HV_参考点1.1': equal_mean(own, 'hv_union'), '最大原始HV_参考点1.1': best['hv_union'],
                '平均完整B5000运行秒': None if is_alias else equal_mean(own, 'full_run_seconds'),
                '平均控制器截至节点秒': None if is_alias else equal_mean(own, 'controller_seconds'),
                '最大HV实例': best['instance'], '最大HV种子': best['seed'],
                '最大提升实例': b['max_percent_instance'], '最大提升种子': b['max_percent_seed'],
                '来源': 'u0与共享同策略复用；2次实际复现' if is_alias else '实际运行'})
    return summary, stats


def csv_bytes(rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return f.getvalue().encode('utf-8-sig')


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def show(table, out):
    def fmt(x, percent=False):
        return '未定义' if x is None else (f'{x:+.2f}%' if percent else f'{x:.6f}')
    for n in BUDGETS:
        print(f'\n===== {n}次评价：同实例跨预算联合边界；HV除以1.331 =====')
        print('方法 | 平均HV | 最大HV | 平均配对提升Base | 最大配对提升Base')
        for r in table:
            if r['评价次数'] == n:
                print(' | '.join([r['方法'], fmt(r['平均HV_0至1']), fmt(r['单次最大HV_0至1']),
                                  fmt(r['平均配对提升Base(%)'], True), fmt(r['单次最大配对提升Base(%)'], True)]))
    print(f'\n[HV-UNION] COMPLETE; zero new evaluations; {out}', flush=True)


def execute(source, stage='auto', out=None, draws=10000):
    source = Path(source).resolve()
    inputs = {}
    def load(path):
        content = path.read_bytes()
        inputs[str(path)] = sha(content)
        return json.loads(content)
    if stage == 'auto':
        stage = 'supervised_results'
        final = source/'results'/'QUICK_RESULT.json'
        if final.exists() and json.loads(final.read_bytes()).get('completed'):
            stage = 'results'
    plan = load(source/'PLAN.json')
    report = load(source/stage/'QUICK_RESULT.json')
    physical = load(source/stage/'PAIRED_RUNS.json')
    require(draws >= 1000, 'Use at least 1000 bootstrap draws')
    rows, methods = validate_panel(physical, plan, report)
    require(('ICM_TERMINAL_PPO' in methods) == (stage == 'results'), 'Report stage/method mismatch')
    destination = Path(out).resolve() if out else source.with_name(source.name+'_hv_union_common_v1')/stage
    require(destination != source and source not in destination.parents and destination not in source.parents,
            'Output must be outside the live experiment directory')
    manifest = dict(inputs=inputs, code_sha256=sha(Path(__file__).read_bytes()), stage=stage,
                    bounds='per instance; all methods, all seeds, all 400/1000/5000 feasible fronts',
                    reference=list(REF), hv01_divisor=VOLUME, bootstrap_draws=draws, bootstrap_seed=953001)
    if destination.exists():
        saved = json.loads((destination/'QUICK_RESULT.json').read_bytes())
        require(saved['manifest'] == manifest and saved['completed'], 'Output already exists with different inputs/settings')
        for name, digest in saved['output_sha256'].items():
            require(sha((destination/name).read_bytes()) == digest, 'Existing output changed: '+name)
        show(saved['summary'], destination)
        return destination
    print(f'[HV-UNION] {stage}: {len(physical)} physical runs, {len(rows)} reported rows; {len(methods)} methods', flush=True)
    if stage == 'supervised_results':
        print('[HV-UNION] PPO final comparison is not included yet; this is the complete 10-method report.', flush=True)
    per_run, bounds = renormalize(rows)
    summary, stats = tables(per_run, methods, report.get('ppo_alias'), draws)
    # Read files once for a coherent snapshot, then verify they were not replaced during calculation.
    require(all(sha(Path(p).read_bytes()) == h for p, h in inputs.items()), 'Source changed during postprocessing; rerun')
    payload = {'00_平均值和最大值.csv': csv_bytes(summary), 'STATISTICS.csv': csv_bytes(stats),
               'PER_RUN.csv': csv_bytes(per_run), 'INSTANCE_BOUNDS.json': json_bytes(bounds)}
    result = dict(completed=True, stage=stage, all_methods_including_ppo_complete=stage == 'results',
        manifest=manifest, methods=methods, instances=len(plan['screen']), seeds=plan['settings']['screen_seeds'],
        physical_runs=len(physical), reported_runs=len(rows), ppo_alias=report.get('ppo_alias'),
        new_searches=0, new_objective_evaluations=0, summary=summary,
        qualification='Post-hoc development reporting only. Training/reward normalization unchanged. Configuration-equal paired means. Fixed configurations; bootstrap instances then seeds within configuration. One instance/configuration limits generalization. Maxima are descriptive, not a replacement for means. Zero reference HV makes percentage summary undefined. Bounds must be recomputed for every method if the included method set changes.',
        output_sha256={name: sha(data) for name, data in payload.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hv-union-', dir=destination.parent) as tmp:
        folder = Path(tmp)/'report'
        folder.mkdir()
        for name, data in payload.items():
            (folder/name).write_bytes(data)
        (folder/'QUICK_RESULT.json').write_bytes(json_bytes(result))
        folder.rename(destination)
    show(summary, destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', choices=['auto', 'supervised_results', 'results'], default='auto')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    execute(args.source, args.stage, args.out)
