"""Joint I -> C preview -> one actual M step, with a single real-evaluation counter."""
from bridge import *
from collection import FullBudgetController
from copy import deepcopy
import random
import time
import resource
import socket
import baseline_uncertainty as base
from crossover import _pair_proxy_rows,_pair_batch_tokens,_pair_candidate_features
from transformer_ppo.episode import run_episode,_optimisation_vector
from policy import TerminalPolicy


class JointController(FullBudgetController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        joint=self.arm!='BASE'
        self.controls_initialization=self.controls_crossover=self.controls_mutation=joint
        self.deferred_pairs=0

    def _select(self,component,row):
        stats=self.selection[component];stats['calls']+=1;chosen=0
        if self.learned is not None:
            started=time.perf_counter();chosen=int(self.learned.select(row));self.inference_seconds+=time.perf_counter()-started
            stats['fallbacks']+=int(chosen==0)
        if not row['legal_mask'][chosen]:raise ValueError('Selector returned invalid candidate')
        stats['switches']+=int(chosen!=0);return chosen

    def mutate(self,*args,**kwargs):
        before=self.arm
        self.arm='BASE' if before=='BASE' else ('M_LEARN' if self.learned is not None else 'M_RULE')
        try:return super().mutate(*args,**kwargs)
        finally:self.arm=before

    def process_crossover_pair(self,function,c1,c2,batches,path_lib,tt_dict,arc_lookup,arcs,wcost,wemis,
                               reliable_options=None,paired_mutation_rate=None,**kwargs):
        active=self.pending_c;self.pending_c=None
        if not active or c1 is not active['pairs'][0][0] or c2 is not active['pairs'][0][1]:return None
        started=time.perf_counter();pairs=active['pairs'];before_pairs=deepcopy(pairs)
        py=random.getstate();nps=np.random.get_state()
        try:
            # C inputs match supervision: Base-mutated complete pairs. Preview
            # consumes no objective evaluation. Restore the stream afterwards.
            for pos in (0,1):self.ch._paired_base_mutation(pairs,pos,batches,path_lib,tt_dict,arc_lookup,reliable_options,paired_mutation_rate)
            for pair in pairs:
                for child in pair:base.repair_missing_allocations(child,batches,path_lib)
        finally:random.setstate(py);np.random.set_state(nps)
        proxy=_pair_proxy_rows(pairs,batches,arcs)
        ref=min(range(8),key=lambda i:(proxy[i]['score'],i!=1,i))
        _,row=self._state('C',pairs,_pair_candidate_features(proxy,active['roles']),[_pair_batch_tokens(p,batches) for p in pairs],ref)
        self._feature_seconds+=time.perf_counter()-started
        selected=self._select('C',row);order=[ref]+[i for i in range(8) if i!=ref]
        for target,source in zip((c1,c2),before_pairs[order[selected]]):
            target.__dict__.clear();target.__dict__.update(deepcopy(source.__dict__))
        self.deferred_pairs+=1
        # NSGA-II now makes its normal mutation coins and calls M once per
        # eligible child. Only these final children enter the real evaluator.
        return None

    def summary(self):return {**super().summary(),'deferred_c_pairs':self.deferred_pairs}


def run_search(instance,settings,task,out,plan_sha):
    torch.set_num_threads(1);info,_=read_instance(instance)
    expected_role='training' if task['sample'] else ('validation' if task['phase']=='validation' else 'screen')
    if info['development_role']!=expected_role:raise ValueError('Wrong instance role for task')
    arm=task['arm'];model=Path(task['model']) if task.get('model') else None
    if model and digest(model/'BUNDLE.json')!=task['bundle_sha256']:raise ValueError('Task model changed')
    policy=None
    if arm=='ICM_INDEPENDENT':policy=IndependentPolicy(model)
    elif arm=='ICM_SHARED_SUPERVISED':policy=SharedPolicy(model)
    elif arm=='ICM_TERMINAL_PPO':policy=TerminalPolicy(model,sample=task['sample'],seed=task['seed']+7000000,record=task['sample'])
    holders=[]
    def factory(*args,**kwargs):
        ctl=JointController(*args,settings=settings,arm=arm,learned=policy,**kwargs);holders.append(ctl);return ctl
    normalizer=legacy_learning.normalizer(task['normalization'])
    old_clear=base.clear_inherited_evaluation_cache;base.clear_inherited_evaluation_cache=lambda *args:None
    try:
        result=run_episode(EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',instance,out,None,normalizer,
            algorithm_seed=task['seed'],evaluation_budget=settings['budget'],pop_size=settings['population'],generations=1000000,
            evaluation_mode='ccp',component='BASE',scenario_count=settings['scenario_count'],ccp_confidence=settings['confidence'],
            uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=arm)
    finally:base.clear_inherited_evaluation_cache=old_clear
    if not settings['budget']-1<=result['real_objective_evaluation_count']<=settings['budget']:
        raise ValueError('Run stopped before its evaluation cap')
    if result['hidden_candidate_objective_evaluations']!=0:raise ValueError('Hidden candidate evaluations')
    points=[list(_optimisation_vector(r)) for r in read_json(out/'final_archive_lineage.json')]
    trajectory=None
    if task['sample']:
        policy.save_events(out/'TRAJECTORY.npz');trajectory=digest(out/'TRAJECTORY.npz')
    result.update(instance=info['instance_id'],configuration=info['configuration'],instance_digest=info['instance_digest'],
        seed=task['seed'],arm=arm,phase=task['phase'],stochastic=task['sample'],plan_sha256=plan_sha,
        task_sha256=canonical_digest(task),model_bundle_sha256=task.get('bundle_sha256'),trajectory_sha256=trajectory,
        final_front_raw=points,scenario_digest=read_json(out/'configuration.json')['scenario_digest'],
        hostname=socket.gethostname(),peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2))
    result['milestone_hv']=[p for p in result['milestone_hv'] if p['evaluation']!=settings['budget']]
    result['milestone_hv'].append({'evaluation':settings['budget'],'actual_evaluations':result['real_objective_evaluation_count'],'hv':result['final_hypervolume']})
    write_json(out/'RUN_COMPLETE.json',result)
    return result
