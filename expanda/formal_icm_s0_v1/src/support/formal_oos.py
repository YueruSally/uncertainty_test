"""Independent fixed-decision CCP5000 re-evaluation; never repair or search."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import csv
import math
import os
import subprocess
import sys
import threading
import time
import numpy as np
import baseline_uncertainty as base
from runtime_common import read_json, write_json, digest, _normalizer, paired_summary
from run_ccp_candidate_pool import decision_signature
from run_ev_ccp_oos_pilot import scenario_digest
from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.instance_catalog import read_instance

def restore_individual(row, path_lib, tt_dict=None, arc_lookup=None,
                       restoration_stats=None):
    """Restore the exact persisted decision, including evolved paths.

    Crossover and mode mutation can create valid ``Path`` objects that are not
    members of the capped static path library.  Their complete decision data
    is persisted, so rebuild those paths from the underlying arcs rather than
    substituting a library path.  Road fallback is deliberately disabled:
    restoration must preserve every saved mode exactly.
    """
    lookup={(tuple(p.nodes),tuple(p.modes)):p for paths in path_lib.values() for p in paths}
    ind=base.Individual()
    used_fallback=False
    for block in row["decision"]:
        key=(block["origin"],block["destination"],int(block["batch_id"]))
        restored=[]
        for allocation in block["allocations"]:
            path_key=(tuple(allocation["nodes"]),tuple(allocation["modes"]))
            path=lookup.get(path_key)
            if path is None:
                if tt_dict is None or arc_lookup is None:
                    raise RuntimeError(
                        "saved path is absent from static lookup and exact "
                        "network reconstruction was not configured: "
                        f"{path_key}")
                path=base.rebuild_path_from_nodes_modes(
                    block["origin"],block["destination"],
                    allocation["nodes"],allocation["modes"],
                    tt_dict,arc_lookup,allow_road_fallback=False)
                if path is None or (tuple(path.nodes),tuple(path.modes)) != path_key:
                    raise RuntimeError(
                        "formal-data integrity failure: saved path cannot be "
                        f"reconstructed exactly from underlying arcs: {path_key}")
                lookup[path_key]=path
                used_fallback=True
                if restoration_stats is not None:
                    restoration_stats["fallback_allocations"] = (
                        restoration_stats.get("fallback_allocations",0)+1)
            restored.append(base.PathAllocation(path,float(allocation["share"])))
        ind.od_allocations[key]=restored
    if decision_signature(ind)!=row["decision_fingerprint"]: raise RuntimeError("restored decision fingerprint mismatch")
    if restoration_stats is not None:
        label="fallback_candidates" if used_fallback else "static_candidates"
        restoration_stats[label]=restoration_stats.get(label,0)+1
    return ind



def evaluate_seed(package, out, config, name, seed):
    instance_path = package/'assets/formal_instances'/f'{name}.json'
    instance, batches = read_instance(instance_path)
    data = package/'assets/data/data_expanded.xlsx'
    if digest(data) != instance['source_data_sha256']: raise ValueError('OOS workbook changed')
    plan_sha = digest(out/'PLAN.json')
    target = out/'oos'/name/f'seed_{seed}.json'
    previous = read_json(target) if target.exists() else {}
    if previous and previous.get('plan_sha256') != plan_sha: raise ValueError('OOS plan changed')
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    _, _, hold, proc, trans_cost, arcs, timetables, _, wait_cost, wait_emission, carbon, _, _, trans_map, _, theta, _ = network
    paths = {m: out/'runs'/name/f'seed_{seed}'/m for m in config['methods']}
    reference_config = read_json(paths['BASE']/'configuration.json')
    profile = reference_config['uncertainty']
    border = base.load_border_delay_map_from_workbook(data, include_customs=profile['border_include_customs'])
    scenarios = base.build_scenario_set(arcs, border, config['oos_scenario_count'], instance['oos5000_seed'],
        stochastic=True, border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
        mode_time_cv=profile['mode_time_cv'], mode_time_max_factor=profile['mode_time_max_factor'],
        border_delay_cv=profile['border_delay_cv'], border_delay_max_factor=profile['border_delay_max_factor'],
        border_mean_source=profile['border_mean_source'])
    sha = scenario_digest(scenarios)
    if sha == reference_config['scenario_digest']: raise ValueError('OOS scenarios equal search scenarios')
    base.ACTIVE_MUTATION_LOGGER = None
    base.ACTIVE_SCENARIO_SET = scenarios; base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = 'ccp'
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = config['ccp_confidence']
    normalizer = _normalizer(package/'assets/reward_normalization.json').with_divisor((
        max(1.,sum(b.quantity for b in batches)),max(1.,sum(b.quantity for b in batches)),
        max(1.,max(b.LT for b in batches)-min(b.ET for b in batches))))
    tt=base.build_timetable_dict(timetables); lookup=base.build_arc_lookup(arcs)
    result = previous or {'instance':name,'algorithm_seed':seed,'plan_sha256':plan_sha,
        'scenario_digest':sha,'scenario_count':scenarios.size,'reference':list(normalizer.reference),'methods':{},'completed':False}
    if result['scenario_digest'] != sha: raise ValueError('OOS scenario cache changed')
    cache = {}
    for method, folder in paths.items():
        front_file = folder/'final_feasible_nondominated.json'
        cfg_file = folder/'configuration.json'
        hashes = {'front':digest(front_file),'configuration':digest(cfg_file)}
        if method in result['methods']:
            if result['methods'][method]['source_sha256'] != hashes: raise ValueError('Saved front changed')
            continue
        cfg=read_json(cfg_file)
        if cfg['instance_digest'] != instance['instance_digest'] or cfg['scenario_digest'] != reference_config['scenario_digest']:
            raise ValueError('OOS source instance/scenario pairing changed')
        for key in ('name','mode_time_cv','mode_time_max_factor','border_delay_cv','border_delay_max_factor','border_mean_source','border_include_customs'):
            if cfg['uncertainty'][key] != profile[key]:raise ValueError('OOS uncertainty model changed')
        front=read_json(front_file); records=[]; points=[]; new_calls=0
        started=time.monotonic()
        for row in front:
            fp=row['decision_fingerprint']
            if fp not in cache:
                ind=restore_individual(row,{},tt,lookup)
                base.evaluate_individual(ind,batches,arcs,tt,wait_cost,wait_emission,
                    node_hold_cost=hold,node_proc_cost=proc,carbon_tax_map=carbon,
                    trans_map=trans_map,border_delay_map=border,theta_rm=theta,node_trans_cost=trans_cost)
                if decision_signature(ind)!=fp: raise ValueError('OOS changed a fixed decision')
                objectives=[float(x) for x in ind.objectives]
                finite=all(math.isfinite(x) for x in objectives)
                if ind.feasible and not finite: raise ValueError('Feasible solution has nonfinite OOS objectives')
                cache[fp]={'decision_fingerprint':fp,'feasible':bool(ind.feasible),'objectives':objectives if finite else None,
                           'normalized_violation':float(ind.normalized_violation) if math.isfinite(ind.normalized_violation) else None}
                new_calls+=1
            item=cache[fp];records.append(item)
            if item['feasible']:points.append(list(normalizer.transform(item['objectives'])))
        result['methods'][method]={'hv':exact_hv_3d(points,normalizer.reference),'source_front_size':len(front),
            'feasible_count':sum(x['feasible'] for x in records),'normalized_feasible_points':points,
            'actual_objective_evaluations':new_calls,'logical_front_evaluations':len(front),
            'scenario_evaluations':new_calls*scenarios.size,'seconds':time.monotonic()-started,
            'source_sha256':hashes,'candidates':records}
        write_json(target,result)
        print(f'[FORMAL-OOS] {name}/{seed}/{method} HV={result["methods"][method]["hv"]:.7g}',flush=True)
    result['completed']=True;write_json(target,result)
    return result


def summarize_oos(rows, config):
    tables=[]
    references={'I_K8_RESIDUAL':'I_DIRECTED_BALANCED','C_K8_RESIDUAL':'C_K8_SIMPLE','M_K8_TRANSFORMER':'M_BALANCED'}
    comparisons={}
    for method in config['methods']:
        table={'method':method}
        refs = [('BASE','vs_base')]+([(references[method],'vs_rule')] if method in references else [])
        if method == 'I_C_M_K8_TRANSFORMER':
            refs.extend((m,'vs_'+m.lower()) for m in references)
        for reference,label in refs:
            records=[]
            for row in rows:
                value=row['methods'][method]['hv']; ref=row['methods'][reference]['hv']
                records.append({'instance':row['instance'],'delta':value-ref,'percent':100*(value/ref-1) if ref>1e-15 else None})
            absolute=paired_summary(records,'delta',941100+config['methods'].index(method))
            relative=paired_summary(records,'percent',941200+config['methods'].index(method))
            comparisons[method+'_'+label]={'absolute_hv_gain':absolute,'percent_gain':relative}
            table['mean_absolute_hv_'+label]=absolute['mean']
            table['mean_percent_'+label]=relative['mean']
            table['undefined_percent_pairs_'+label]=relative.get('undefined_pairs',0)
        table['mean_final_hv'] = float(np.mean([r['methods'][method]['hv'] for r in rows]))
        table['feasible_front_solutions']=sum(r['methods'][method]['feasible_count'] for r in rows)
        table['total_front_solutions']=sum(r['methods'][method]['source_front_size'] for r in rows)
        tables.append(table)
    return {'completed':True,'table':tables,'comparisons':comparisons,'reference_rule':config['oos_reference'],
        'scenario_count':config['oos_scenario_count'],'new_searches':0,'new_training_steps':0,
        'actual_oos_objective_evaluations':sum(m['actual_objective_evaluations'] for r in rows for m in r['methods'].values()),
        'logical_oos_front_evaluations':sum(m['logical_front_evaluations'] for r in rows for m in r['methods'].values()),
        'qualification':'fixed decisions re-evaluated without repair; infeasible solutions excluded from HV; zero reference HV makes percent undefined, never omitted from absolute comparison'}


def run_oos(package,out,config,config_path):
    if not (out/'SEARCH_COMPLETE').exists():raise ValueError('Complete the frozen search grid before OOS')
    if (out/'FORMAL_COMPLETE').exists():
        print('[FORMAL] Already complete; see FINAL_REPORT.json',flush=True);return
    catalog=read_json(package/'assets/formal_instances/catalog.json')
    tasks=[(r['instance_id'],s) for r in catalog['instances'] for s in config['algorithm_seeds']]
    status={'stage':'oos','completed_pairs':0,'total_pairs':len(tasks)}
    lock=threading.Lock();stop=threading.Event();deadline=time.monotonic()+config['invocation_limit_seconds']
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    def task(name,seed):
        if stop.is_set() or time.monotonic() >= deadline: raise TimeoutError('OOS invocation stopped; resume with --run')
        log=out/'logs'/f'oos_{name}_{seed}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(package/'run.py'),'--config',str(config_path),'--out',str(out),'--oos-worker','--instance',name,'--seed',str(seed)]
        with log.open('w') as f,subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env) as process:
            began=time.monotonic()
            while process.poll() is None:
                if stop.is_set() or time.monotonic()>=deadline or time.monotonic()-began>=config['per_run_timeout_seconds']:
                    process.kill();process.wait();raise TimeoutError('OOS interrupted; resumable: '+str(log))
                time.sleep(.2)
            if process.returncode:raise RuntimeError(str(log)+'\n'+log.read_text(errors='replace')[-2500:])
        value=read_json(out/'oos'/name/f'seed_{seed}.json')
        if not value['completed']:raise ValueError('Incomplete OOS pair')
        with lock:
            status['completed_pairs']+=1;write_json(out/'OOS_STATUS.json',status)
            print(f'[FORMAL-OOS] {status["completed_pairs"]}/{len(tasks)} pairs complete',flush=True)
        return value
    rows=[]
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures=[pool.submit(task,*t) for t in tasks]
        try:
            for future in as_completed(futures):rows.append(future.result())
        except BaseException as exc:
            write_json(out/'OOS_INTERRUPTED.json', {'error':str(exc),'uncommitted_attempt_cost_may_be_missing':True})
            stop.set()
            for f in futures:f.cancel()
            raise
    rows.sort(key=lambda r:(r['instance'],r['algorithm_seed']))
    report=summarize_oos(rows,config)
    report['uncommitted_oos_attempt_cost_not_in_total'] = (out/'OOS_INTERRUPTED.json').exists()
    write_json(out/'OOS_RESULT.json',report)
    fields=list(dict.fromkeys(k for row in report['table'] for k in row))
    with (out/'OOS_RESULT_TABLE.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(report['table'])
    write_json(out/'FINAL_REPORT.json',{'completed':True,'search':read_json(out/'QUICK_RESULT.json'),'independent_scenarios':report,
        'final_test_opened':False,'ppo_updates':0,'role':'formal S0 component selection at target budget',
        'next_action':'review paired estimates, intervals, independent scenarios and time before selecting a component; no automatic retraining'})
    write_json(out/'OOS_STATUS.json',{**status,'stage':'complete'})
    (out/'FORMAL_COMPLETE').write_text(digest(out/'FINAL_REPORT.json')+'\n')
    print('[FORMAL] COMPLETE search and independent-scenario re-evaluation',flush=True)
