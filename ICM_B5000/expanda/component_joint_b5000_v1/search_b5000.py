"""Same frozen engine and models, with isolated component hooks and saved fronts."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'joint_icm_multiscale_terminal_v1'))
from bridge import *
from search import JointController
from controller import ScreenController
from policy import TerminalPolicy
from transformer_ppo.episode import run_episode,_optimisation_vector
from transformer_ppo.archive import nondominated
import baseline_uncertainty as base
from copy import deepcopy
import resource
import socket
import time

SINGLES = ('I_RULE','I_LEARN','C_RULE','C_LEARN','M_RULE','M_LEARN')
ARMS = ('BASE',)+SINGLES+('ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED','ICM_TERMINAL_PPO')


def enabled_components(arm):
    if arm == 'BASE': return ()
    if arm in SINGLES: return (arm[0],)
    if arm in ARMS: return tuple('ICM')
    raise ValueError('Unknown method: '+arm)


class BudgetController(JointController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        active=enabled_components(self.arm)
        self.controls_initialization='I' in active
        self.controls_crossover='C' in active
        self.controls_mutation='M' in active
        self.budget_fronts=[]

    def process_crossover_pair(self,*args,**kwargs):
        if self.arm in ('C_RULE','C_LEARN'):
            # Isolated C deploys exactly its supervised unit: C + Base mutation.
            return ScreenController.process_crossover_pair(self,*args,**kwargs)
        return super().process_crossover_pair(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        before=len(self.checkpoints)
        value=super().evaluate(*args,**kwargs)
        if len(self.checkpoints)>before:
            candidates=deepcopy(self.population+self.pending_children)
            selected=base.nsga2_environmental_selection(candidates,min(len(candidates),self.settings['population']))
            self.budget_fronts.append(dict(evaluation=self.evaluations,actual_evaluations=self.evaluations,
                hv=self.checkpoints[-1]['hv'],front_raw=nondominated([list(x.objectives) for x in selected if x.feasible]),
                controller_seconds=time.perf_counter()-self.start,
                definition='read-only environmental selection over evaluated population and pending children'))
        return value


def run_search(instance,settings,task,out,plan_sha):
    torch.set_num_threads(1);info,_=read_instance(instance)
    expected='training' if task['sample'] else ('validation' if task['phase']=='validation' else 'screen')
    if info['development_role']!=expected:raise ValueError('Wrong instance role')
    arm=task['arm'];enabled_components(arm)
    model=Path(task['model']) if task.get('model') else None
    if model and digest(model/'BUNDLE.json')!=task['bundle_sha256']:raise ValueError('Model changed')
    policy=None
    if arm.endswith('_LEARN') or arm=='ICM_INDEPENDENT':policy=IndependentPolicy(model)
    elif arm=='ICM_SHARED_SUPERVISED':policy=SharedPolicy(model)
    elif arm=='ICM_TERMINAL_PPO':policy=TerminalPolicy(model,sample=task['sample'],seed=task['seed']+7000000,record=task['sample'])
    holders=[]
    def factory(*args,**kwargs):
        ctl=BudgetController(*args,settings=settings,arm=arm,learned=policy,**kwargs);holders.append(ctl);return ctl
    normalizer=legacy_learning.normalizer(task['normalization'])
    old=base.clear_inherited_evaluation_cache;base.clear_inherited_evaluation_cache=lambda *args:None
    try:
        result=run_episode(EXPANDA/'formal_icm_s0_v1/assets/data/data_expanded.xlsx',instance,out,None,normalizer,
            algorithm_seed=task['seed'],evaluation_budget=settings['budget'],pop_size=settings['population'],generations=1000000,
            evaluation_mode='ccp',component='BASE',scenario_count=settings['scenario_count'],ccp_confidence=settings['confidence'],
            uncertainty_profile='transport_border_saa_v1',inference_controller_factories={'BASE':factory},method_override=arm)
    finally:base.clear_inherited_evaluation_cache=old
    if not settings['budget']-1<=result['real_objective_evaluation_count']<=settings['budget']:raise ValueError('Incomplete budget')
    if result['hidden_candidate_objective_evaluations']!=0:raise ValueError('Hidden evaluation')
    ctl=holders[0];points=[list(_optimisation_vector(r)) for r in read_json(out/'final_archive_lineage.json')]
    milestones=[p for p in ctl.budget_fronts if p['evaluation']!=settings['budget']]
    milestones.append(dict(evaluation=settings['budget'],actual_evaluations=result['real_objective_evaluation_count'],
        hv=result['final_hypervolume'],front_raw=points,controller_seconds=time.perf_counter()-ctl.start,definition='actual final feasible nondominated front'))
    if {p['evaluation'] for p in milestones}!=set(settings['milestones']):raise ValueError('Missing budget snapshot')
    trajectory=None
    if task['sample']:
        policy.save_events(out/'TRAJECTORY.npz');trajectory=digest(out/'TRAJECTORY.npz')
    result.update(instance=info['instance_id'],configuration=info['configuration'],instance_digest=info['instance_digest'],seed=task['seed'],
        arm=arm,phase=task['phase'],stochastic=task['sample'],plan_sha256=plan_sha,task_sha256=canonical_digest(task),
        model_bundle_sha256=task.get('bundle_sha256'),trajectory_sha256=trajectory,final_front_raw=points,
        scenario_digest=read_json(out/'configuration.json')['scenario_digest'],enabled_components=list(enabled_components(arm)),
        budget_fronts=milestones,milestone_hv=[{k:p[k] for k in ('evaluation','actual_evaluations','hv')} for p in milestones],
        hostname=socket.gethostname(),peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform!='darwin' else 1024**2))
    for c in 'ICM':
        if c not in enabled_components(arm) and result['selection'][c]['calls']:
            raise ValueError('Inactive component made choices')
    write_json(out/'RUN_COMPLETE.json',result);return result
