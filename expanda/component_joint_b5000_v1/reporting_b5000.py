"""All budgets use fixed training normalization; union results stay separate."""
from search_b5000 import *
from reporting import csv_write,union_hv
from report import comparison

NAMES={'BASE':'Base','I_RULE':'I 启发式','I_LEARN':'I Transformer','C_RULE':'C 启发式','C_LEARN':'C Transformer',
       'M_RULE':'M 启发式','M_LEARN':'M Transformer','ICM_HEURISTIC':'ICM 启发式','ICM_INDEPENDENT':'ICM 独立 Transformer',
       'ICM_SHARED_SUPERVISED':'ICM 共享监督 Transformer','ICM_TERMINAL_PPO':'ICM 终局联合 PPO'}


def reference(arm):
    if arm in SINGLES:return arm[0]+'_RULE'
    if arm=='BASE':return 'BASE'
    return 'ICM_HEURISTIC'


def report(physical,plan,out,partial=False,reuse=None):
    settings=plan['settings'];rows=[r for r in physical if r['arm']!='ICM_TERMINAL_PPO' or not reuse]
    methods=[m for m in ARMS if any(r['arm']==m for r in rows)]
    expected_methods=ARMS if (reuse or 'ICM_TERMINAL_PPO' in methods) else ARMS[:-1]
    if reuse:
        rows += [dict(r,arm='ICM_TERMINAL_PPO',reported_alias_of='ICM_SHARED_SUPERVISED') for r in rows if r['arm']=='ICM_SHARED_SUPERVISED']
        methods=list(ARMS)
    if tuple(methods)!=tuple(expected_methods):raise ValueError('Incomplete method block')
    keys={(r['instance'],r['seed'],r['arm']) for r in rows}
    if len(keys)!=len(rows):raise ValueError('Duplicate run')
    if not partial:
        expected={(r['instance_id'],seed,m) for r in plan['screen'] for seed in settings['screen_seeds'] for m in methods}
        if keys!=expected:raise ValueError('Incomplete panel')
    groups={}
    for r in rows:
        if (r['plan_sha256']!=plan['sha256'] or r['phase']!='screen' or r['stochastic'] or
            r['hidden_candidate_objective_evaluations'] or r['ppo_updates'] or r['evaluation_budget']!=settings['budget'] or
            not settings['budget']-1<=r['real_objective_evaluation_count']<=settings['budget']):raise ValueError('Invalid frozen result')
        groups.setdefault((r['instance'],r['seed']),[]).append(r)
    for group in groups.values():
        if {r['arm'] for r in group}!=set(methods) or len({r['scenario_digest'] for r in group})!=1 or len({r['instance_digest'] for r in group})!=1:
            raise ValueError('Pairing failed')
    out.mkdir(parents=True,exist_ok=True);main=[];secondary=[];stats=[];per=[];bounds_all={};quick={};per_run=[]
    for evaluation in settings['milestones']:
        sample=[]
        for r in rows:
            found=[p for p in r['budget_fronts'] if p['evaluation']==evaluation]
            if len(found)!=1:raise ValueError('Missing/duplicated snapshot')
            p=found[0]
            sample.append(dict(r,final_hypervolume=p['hv'],final_front_raw=p['front_raw'],
                              recorded_evaluations=p['actual_evaluations'],controller_seconds=p['controller_seconds']))
        # All rows, including Base and aliases, use precisely the same per-instance bounds.
        union,bounds=union_hv(sample);bounds_all[str(evaluation)]=bounds;quick[str(evaluation)]={}
        union_index={(r['instance'],r['seed'],r['arm']):r['final_hypervolume'] for r in union}
        for r in physical:
            p=next(p for p in r['budget_fronts'] if p['evaluation']==evaluation)
            per_run.append(dict(instance=r['instance'],configuration=r['configuration'],seed=r['seed'],arm=r['arm'],
                evaluation=evaluation,actual_evaluations=p['actual_evaluations'],hv_frozen_training=p['hv'],
                hv_instance_union_at_budget=union_index[r['instance'],r['seed'],r['arm']],
                controller_seconds=p['controller_seconds'],full_run_seconds=r['optimisation_seconds'] if evaluation==settings['budget'] else None,
                instance_digest=r['instance_digest'],scenario_digest=r['scenario_digest']))
        for label,values,table in [('frozen_training',sample,main),('instance_union_at_budget',union,secondary)]:
            for m in methods:
                own=[r for r in values if r['arm']==m];ref=reference(m)
                refs=list(dict.fromkeys(('BASE',ref,'ICM_INDEPENDENT','ICM_SHARED_SUPERVISED')))
                comparisons={r:comparison(values,m,r) for r in refs}
                b=comparisons['BASE'];h=comparisons[ref];alias=bool(reuse and m=='ICM_TERMINAL_PPO')
                table.append({'评价上限':evaluation,'方法':NAMES[m],'平均HV':float(np.mean([r['final_hypervolume'] for r in own])),
                    '相对Base提升(%)':b['percent']['mean'],'Base区间下限(%)':b['percent']['low'],'Base区间上限(%)':b['percent']['high'],
                    '规则对照':NAMES[ref],'相对规则提升(%)':h['percent']['mean'],'规则区间下限(%)':h['percent']['low'],'规则区间上限(%)':h['percent']['high'],
                    '胜Base':b['wins'],'平Base':b['ties'],'负Base':b['losses'],
                    '控制器运行至该点秒':None if alias else float(np.mean([r['controller_seconds'] for r in own])),
                    '完整B5000搜索秒':None if alias or evaluation!=settings['budget'] else float(np.mean([r['optimisation_seconds'] for r in own])),
                    '结果来源':'u0同策略复用；非独立组' if alias else '实际运行',
                    '归一化':label})
                if label=='frozen_training':
                    quick[str(evaluation)][m]=comparisons
                    per.extend(dict(evaluation=evaluation,method=m,**r) for r in b['per_configuration'])
                for reference_name,v in comparisons.items():
                    stats.append(dict(evaluation=evaluation,normalization=label,method=m,reference=reference_name,
                        mean_percent=v['percent']['mean'],ci_low=v['percent']['low'],ci_high=v['percent']['high'],
                        absolute_difference=v['absolute']['mean'],absolute_ci_low=v['absolute']['low'],absolute_ci_high=v['absolute']['high'],
                        wins=v['wins'],ties=v['ties'],losses=v['losses'],zero_reference_pairs=v['reference_zero_pairs'],alias=alias))
    csv_write(out/'00_全部方法_冻结归一化.csv',main)
    csv_write(out/'01_全部方法_逐预算联合归一化.csv',secondary)
    csv_write(out/'STATISTICS.csv',stats);csv_write(out/'PER_CONFIGURATION.csv',per)
    csv_write(out/'PER_RUN.csv',per_run)
    write_json(out/'INSTANCE_BOUNDS.json',bounds_all)
    write_json(out/'PAIRED_RUNS.json',physical)
    q=dict(completed=not partial,preview_only=partial,settings=settings,comparisons=quick,ppo_alias=reuse,
        physically_completed_screen_searches=len(physical),screen_evaluations=sum(r['real_objective_evaluation_count'] for r in physical),
        selected_ppo_update=read_json(out.parent/'terminal/SELECTED.json')['update'] if (out.parent/'terminal/SELECTED.json').exists() else None,
        final_test_opened=False,
        qualification='Post-hoc B5000 development version; same previously inspected panel. Fixed training normalization is main. Per-budget union reference changes across budgets; do not use that table as one fixed-reference convergence curve. Intervals conditional on one instance/configuration. Preview is not component selection.')
    write_json(out/'QUICK_RESULT.json',q)
    print('[B5000] report '+str(out),flush=True)
    return q
