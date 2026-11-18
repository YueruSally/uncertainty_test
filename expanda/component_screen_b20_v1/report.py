"""Paired, configuration-equal descriptive selection; no screen-driven refitting."""
from common import *
import csv
from collections import defaultdict
import numpy as np

def comparison(rows,method,reference,bootstrap=10000):
    indexed={(r['configuration'],r['instance'],r['seed'],r['arm']):r for r in rows}
    groups=defaultdict(lambda:defaultdict(list));absolute=defaultdict(lambda:defaultdict(list));wins=ties=losses=zero=0
    for r in rows:
        if r['arm']!=method:continue
        c,i,s=r['configuration'],r['instance'],r['seed'];b=indexed[c,i,s,reference]
        a,h=float(r['final_hypervolume']),float(b['final_hypervolume']);d=a-h
        absolute[c][i].append(d);groups[c][i].append(100*d/h if h>0 else None)
        wins+=int(d>1e-12);losses+=int(d < -1e-12);ties+=int(abs(d)<=1e-12);zero+=int(h==0)
    def summarize(group):
        if any(x is None for v in group.values() for xs in v.values() for x in xs):return {'mean':None,'low':None,'high':None}
        rng=np.random.default_rng(953001);draws=[];means=[]
        for c,inst in sorted(group.items()):
            x=np.asarray([inst[i] for i in sorted(inst)],dtype=float)
            # Each configuration stays in every draw. Resample instances, then
            # seeds within each sampled instance, keeping method pairs intact.
            ids=rng.integers(len(x),size=(bootstrap,len(x)))
            seeds=rng.integers(x.shape[1],size=(bootstrap,len(x),x.shape[1]))
            draws.append(x[ids[:,:,None],seeds].mean(axis=(1,2)));means.append(x.mean())
        sample=np.mean(draws,axis=0)
        return {'mean':float(np.mean(means)),'low':float(np.quantile(sample,.025)),'high':float(np.quantile(sample,.975))}
    per=[]
    for c in sorted(groups):
        vals=[x for a in groups[c].values() for x in a]
        per.append({'configuration':c,'mean_percent':None if any(x is None for x in vals) else float(np.mean(vals)),
                    'mean_absolute_difference':float(np.mean([np.mean(x) for x in absolute[c].values()]))})
    return {'percent':summarize(groups),'absolute':summarize(absolute),'reference_zero_pairs':zero,
            'wins':wins,'ties':ties,'losses':losses,'per_configuration':per}

def csv_write(path,records):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def report(rows,settings,out):
    expected=len(settings['configurations'])*settings['screen_per_configuration']*len(settings['algorithm_seeds'])*len(METHODS)
    if len(rows)!=expected:raise ValueError(f'Incomplete screen {len(rows)}/{expected}')
    pairs=defaultdict(list)
    for r in rows:pairs[r['instance'],r['seed']].append(r)
    for group in pairs.values():
        if {r['arm'] for r in group}!=set(METHODS) or len(group)!=len(METHODS):raise ValueError('Missing/duplicate method')
        if len({r['scenario_digest'] for r in group})!=1:raise ValueError('Paired scenarios differ')
        if any(r['evaluation_budget']!=settings['budget'] or r['hidden_candidate_objective_evaluations']!=0 or r['ppo_updates']!=0 for r in group):raise ValueError('Inconsistent budget or hidden evaluation')
    result={};table=[];per=[]
    for m in METHODS:
        own=[r for r in rows if r['arm']==m];ref=m[0]+'_RULE' if m.endswith('LEARN') else 'BASE'
        b=comparison(rows,m,'BASE');h=comparison(rows,m,ref)
        seconds=float(np.mean([r['optimisation_seconds'] for r in own]))
        result[m]={'vs_base':b,'vs_rule':h,'rule_reference':ref,'mean_seconds':seconds,
            'evaluations':sum(r['real_objective_evaluation_count'] for r in own),'mean_feasible_fraction':float(np.mean([r['final_feasible_fraction'] for r in own])),
            'selection':{k:{v:sum(r['selection'][k][v] for r in own) for v in ('calls','fallbacks','switches','unsupported')} for k in 'ICM'}}
        table.append({'method':m,'percent_vs_base':b['percent']['mean'],'ci_low':b['percent']['low'],'ci_high':b['percent']['high'],
            'absolute_hv_vs_base':b['absolute']['mean'],'rule_reference':ref,'percent_vs_rule':h['percent']['mean'],
            'rule_ci_low':h['percent']['low'],'rule_ci_high':h['percent']['high'],'wins_vs_base':b['wins'],'ties_vs_base':b['ties'],'losses_vs_base':b['losses'],
            'mean_seconds':seconds,'mean_feasible_fraction':result[m]['mean_feasible_fraction']})
        per.extend({'method':m,**x} for x in b['per_configuration'])
    absolute=any(r['final_hypervolume']==0 for r in rows if r['arm']=='BASE');metric='absolute' if absolute else 'percent'
    ranking=sorted(('I_LEARN','C_LEARN','M_LEARN'),key=lambda m:(-result[m]['vs_base'][metric]['mean'],result[m]['mean_seconds']))
    convergence=[]
    for m in METHODS:
        for n in settings['milestones']:
            points=[(r['configuration'],p['hv']) for r in rows if r['arm']==m for p in r['milestone_hv'] if p['evaluation']==n]
            if len(points)!=len(pairs):raise ValueError(f'Missing convergence checkpoints {m}/{n}: {len(points)}')
            convergence.append({'method':m,'evaluation':n,'config_equal_mean_hv':float(np.mean([np.mean([h for c,h in points if c==k]) for k in settings['configurations']]))})
    csv_write(out/'RESULT_TABLE.csv',table);csv_write(out/'PER_CONFIGURATION.csv',per);csv_write(out/'CONVERGENCE.csv',convergence)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for m in METHODS:
        v=[r for r in convergence if r['method']==m];ax.plot([r['evaluation'] for r in v],[r['config_equal_mean_hv'] for r in v],label=m,marker='.')
    ax.set(xlabel='Counted solution evaluations (cap)',ylabel='Configuration-equal normalized feasible HV');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'CONVERGENCE.png',dpi=180);plt.close(fig)
    quick={'completed':True,'reference_method':'BASE','design':settings,'methods':result,'ranking_metric':metric,'component_ranking':ranking,
        'selected_for_future_ppo':ranking[0][0],'qualification':'Development selection at B20000 only; rank is a research choice, not proof of universal superiority. No PPO has run; final test remains closed.',
        'new_ppo_updates':0,'final_test_opened':False,'searches':len(rows),'search_evaluations':sum(r['real_objective_evaluation_count'] for r in rows)}
    write_json(out/'QUICK_RESULT.json',quick);write_json(out/'PAIRED_RUNS.json',rows)
    return quick
