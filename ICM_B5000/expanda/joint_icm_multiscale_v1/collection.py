"""Full-budget labels using unchanged Base C/M trajectories and corrected I prefix."""
from bootstrap import *
import numpy as np
import baseline_uncertainty as base
from controller import ScreenController


def fixed_seed_count(population):
    seeds=int(round(population*base.FEASIBILITY_SEED_FRACTION))
    if base.FEASIBILITY_SEED_FRACTION>0 and seeds==0:seeds=1
    seeds=min(population,max(0,seeds));remaining=population-seeds
    return min(population,seeds+3*(max(1,remaining//6) if remaining else 0))


class FullBudgetController(ScreenController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.i_start_slot=fixed_seed_count(self.settings['population'])
        available=self.settings['population']-self.i_start_slot
        self.i_slots=set(np.linspace(self.i_start_slot,self.settings['population']-1,
                                    min(available,self.settings['states']['I']),dtype=int).tolist())
        self.fixed_slots_preserved=0

    def initial_individual_from_base(self,ind,batches,path_lib,max_paths=3,slot_index=None):
        if int(slot_index)<self.i_start_slot:
            self.slot=int(slot_index);self.fixed_slots_preserved+=1;return ind
        return super().initial_individual_from_base(ind,batches,path_lib,max_paths,slot_index)

    def _state(self,*args,**kwargs):
        pairs,row=super()._state(*args,**kwargs)
        row=progress_row(row,self.settings['progress_denominator'])
        return pairs,row

    def _audit(self,component,*args,**kwargs):
        before=len(self.feedback)
        super()._audit(component,*args,**kwargs)
        if component=='I' and len(self.feedback)>before:
            row=self.feedback[-1];old=row['phase']
            phase=min(2,int(3*(self.slot-self.i_start_slot)/max(1,self.settings['population']-1-self.i_start_slot)))
            self.phase_counts['I'][old]-=1;self.phase_counts['I'][phase]+=1;row['phase']=phase

    def summary(self):
        return {**super().summary(),'fixed_initial_slots_preserved':self.fixed_slots_preserved,
                'progress_denominator':self.settings['progress_denominator']}


def work(root,settings,name,seed,arm,out,sha):
    import worker
    prior,version=worker.ScreenController,worker.FEATURE_VERSION
    worker.ScreenController=FullBudgetController;worker.FEATURE_VERSION=FEATURE
    try:worker.work(root,settings,name,seed,arm,out,sha)
    finally:worker.ScreenController=prior;worker.FEATURE_VERSION=version


def jobs(root,instances):
    return [(r['instance_id'],r['algorithm_seed'],arm,root/'collection'/r['instance_id']/arm)
            for r in instances for arm in ('COLLECT_I','COLLECT_CM')]


def completed_dataset(root,settings,instances,sha):
    from data import validate_rows
    folders=[];runs=[]
    for name,seed,arm,folder in jobs(root,instances):
        done=legacy_runner.completed(folder,sha)
        if not done:raise ValueError('Full-budget collection unfinished: '+str(folder))
        info=next(r for r in instances if r['instance_id']==name)
        d=read_json(folder/'DATA.json')
        if d['role']!=info['development_role'] or d['configuration']!=info['configuration'] or d['instance']!=name:
            raise ValueError('Full collection identity mismatch')
        for c in ('I' if arm=='COLLECT_I' else 'CM'):
            rows=[r for r in d['rows'] if r['component']==c]
            validate_rows(rows,info,c,settings['budget'])
            if not rows or not all(d['phase_counts'][c]):raise ValueError('Missing collection phase: '+name+'/'+c)
            if d['extra_evaluations'][c]>settings['extra_label_budget_per_component_instance']:
                raise ValueError('Extra-label budget exceeded')
        if arm=='COLLECT_CM' and not settings['budget']-1<=done['real_objective_evaluation_count']<=settings['budget']:
            raise ValueError('Incomplete full-budget trajectory')
        if arm=='COLLECT_I' and done['fixed_initial_slots_preserved']!=fixed_seed_count(settings['population']):
            raise ValueError('Fixed initialization seeds were not preserved')
        folders.append(folder);runs.append(done)
    report={'completed':True,'trajectories':len(runs),'base_trajectories':len(instances),
            'initialization_only_trajectories':len(instances),
            'completed_search_evaluations':sum(r['real_objective_evaluation_count'] for r in runs),
            'extra_label_evaluations':{c:sum(r['audit_evaluations'][c] for r in runs) for c in COMPONENTS},
            'search_seconds':sum(r['optimisation_seconds'] for r in runs),
            'states':{c:sum(r['audit_states'][c] for r in runs) for c in COMPONENTS},
            'interrupted_attempts':len(list((root/'interrupted').glob('*'))),
            'qualification':'Completed costs only; interrupted work is preserved separately, not assumed free.'}
    write_json(root/'COLLECTION_SUMMARY.json',report)
    return folders
