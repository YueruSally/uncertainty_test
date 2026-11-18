"""Paired, configuration-equal reports; union bounds are postprocessing only."""
from bridge import *
from collections import defaultdict
import csv
from report import comparison
from transformer_ppo.archive import exact_hv_3d


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def union_hv(raw):
    result=[];bounds={}
    for name in sorted({r['instance'] for r in raw}):
        group=[r for r in raw if r['instance']==name]
        points=np.asarray([p for r in group for p in r['final_front_raw']],dtype=float).reshape(-1,3)
        lo=points.min(0) if len(points) else np.zeros(3);hi=points.max(0) if len(points) else np.zeros(3)
        width=np.where(hi>lo,hi-lo,1.)
        bounds[name]={'minimum':lo.tolist(),'maximum':hi.tolist(),'reference':[1.1]*3,'zero_range':(hi==lo).tolist()}
        for r in group:
            points=np.asarray(r['final_front_raw'],dtype=float).reshape(-1,3)
            result.append({**r,'training_normalization_hv':r['final_hypervolume'],
                           'final_hypervolume':exact_hv_3d(((points-lo)/width).tolist(),[1.1]*3)})
    return result,bounds


def check_pairs(rows,plan):
    s=plan['settings'];expected={(r['instance_id'],seed,m) for r in plan['screen'] for seed in s['screen_seeds'] for m in METHODS}
    if len(rows)!=len(expected) or {(r['instance'],r['seed'],r['arm']) for r in rows}!=expected:raise ValueError('Missing or duplicate screen runs')
    groups=defaultdict(list)
    for r in rows:
        if r['phase']!='screen' or r['stochastic'] or r['hidden_candidate_objective_evaluations']!=0 or r['ppo_updates']!=0:raise ValueError('Screen must be frozen, deterministic and no hidden evaluation')
        if r['evaluation_budget']!=s['budget'] or not s['budget']-1<=r['real_objective_evaluation_count']<=s['budget']:raise ValueError('Different evaluation budgets')
        groups[r['instance'],r['seed']].append(r)
    for group in groups.values():
        if len({r['scenario_digest'] for r in group})!=1 or len({r['instance_digest'] for r in group})!=1:raise ValueError('Broken instance/scenario pairing')


def report(raw,plan,out,selected):
    check_pairs(raw,plan);rows,bounds=union_hv(raw);methods={};table=[];pairs=[];per=[]
    names=dict(zip(METHODS,['Base','ICM 启发式','ICM 独立 Transformer','ICM 共享监督 Transformer','ICM 终局联合 PPO']))
    if selected['update']==0:names['ICM_TERMINAL_PPO']='ICM 终局PPO（验证选择update0，未采用更新）'
    for m in METHODS:
        own=[r for r in rows if r['arm']==m];sec=float(np.mean([r['optimisation_seconds'] for r in own]))
        refs=['BASE','ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED']
        comparisons={ref:comparison(rows,m,ref) for ref in refs}
        b=comparisons['BASE'];rule=comparisons['ICM_HEURISTIC']
        methods[m]={'comparisons':comparisons,'mean_seconds':sec,
                    'selection':{c:{k:sum(r['selection'][c][k] for r in own) for k in ('calls','fallbacks','switches','unsupported')} for c in COMPONENTS}}
        table.append({'方法':names[m],'相对Base提升(%)':b['percent']['mean'],'Base比较95%区间下限(%)':b['percent']['low'],
                      'Base比较95%区间上限(%)':b['percent']['high'],'相对ICM启发式提升(%)':rule['percent']['mean'],
                      '平均搜索时间(秒)':sec,'胜Base':b['wins'],'平Base':b['ties'],'负Base':b['losses']})
        for ref,q in comparisons.items():
            indexed={(r['instance'],r['seed']):r for r in rows if r['arm']==ref}
            deltas=[r['final_hypervolume']-indexed[r['instance'],r['seed']]['final_hypervolume'] for r in own]
            percent=[100*d/indexed[r['instance'],r['seed']]['final_hypervolume'] for r,d in zip(own,deltas) if indexed[r['instance'],r['seed']]['final_hypervolume']>0]
            pairs.append({'method':m,'reference':ref,'pairs':len(own),'mean_percent':q['percent']['mean'],
                'percent_ci_low':q['percent']['low'],'percent_ci_high':q['percent']['high'],
                'mean_absolute':q['absolute']['mean'],'absolute_ci_low':q['absolute']['low'],'absolute_ci_high':q['absolute']['high'],
                'percent_sample_sd':float(np.std(percent,ddof=1)) if len(percent)>1 and len(percent)==len(own) else None,
                'absolute_sample_sd':float(np.std(deltas,ddof=1)) if len(deltas)>1 else None,'wins':q['wins'],'ties':q['ties'],'losses':q['losses'],'zero_reference_pairs':q['reference_zero_pairs']})
        per.extend({'method':m,**r} for r in b['per_configuration'])
    out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'00_联合训练对比.csv',table);csv_write(out/'STATISTICS.csv',pairs);csv_write(out/'PER_CONFIGURATION.csv',per)
    csv_write(out/'PER_RUN.csv',[{k:r[k] for k in ('instance','configuration','seed','arm','final_hypervolume','training_normalization_hv','optimisation_seconds','real_objective_evaluation_count','peak_memory_mib')} for r in rows])
    convergence=[]
    for m in METHODS:
        for e in plan['settings']['milestones']:
            points=[(r['configuration'],p['hv']) for r in raw if r['arm']==m for p in r['milestone_hv'] if p['evaluation']==e]
            if len(points)!=len(plan['screen'])*len(plan['settings']['screen_seeds']):raise ValueError(f'Missing convergence points {m}/{e}')
            convergence.append({'method':m,'evaluation':e,'mean_hv_training_normalization':float(np.mean([np.mean([v for c,v in points if c==cfg]) for cfg in plan['settings']['configurations']]))})
    csv_write(out/'CONVERGENCE.csv',convergence)
    write_json(out/'INSTANCE_BOUNDS.json',bounds);write_json(out/'PAIRED_RUNS.json',rows)
    training_norm={m:{ref:comparison(raw,m,ref) for ref in ('BASE','ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED')} for m in METHODS}
    q={'completed':True,'design':plan['settings'],'selected_ppo_update':selected['update'],'methods':methods,
       'training_normalization_comparisons':training_norm,
       'screen_searches':len(rows),'screen_objective_evaluations':sum(r['real_objective_evaluation_count'] for r in rows),
       'hidden_candidate_objective_evaluations':0,'final_test_opened':False,
       'qualification':'S0-S8 development panel, one new instance/configuration and three seeds. Intervals are conditional on these instances; no within-configuration instance diversity estimated. Union bounds are reporting-only; convergence and PPO selection retain training normalization.'}
    write_json(out/'QUICK_RESULT.json',q);return q
