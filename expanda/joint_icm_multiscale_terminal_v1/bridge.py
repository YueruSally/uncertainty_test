"""New terminal-training experiment; completed supervision remains read-only."""
from pathlib import Path
import sys
HERE2=Path(__file__).resolve().parent
SUP=HERE2.parent/'joint_icm_multiscale_v1'
sys.path.insert(0,str(SUP))
from bootstrap import *
import numpy as np
import torch
import copy
from torch import nn
from shared import SharedTransformer,SharedPolicy,load_checkpoint,save_checkpoint,predictions

METHODS=('BASE','ICM_HEURISTIC','ICM_INDEPENDENT','ICM_SHARED_SUPERVISED','ICM_TERMINAL_PPO')


def files():
    return {**runtime_files(),**{str(p):digest(p) for p in list(HERE2.glob('*.py'))+[HERE2/'protocol.json']}}


class IndependentBank(nn.Module):
    """Same task network architecture but no parameters shared across tasks."""
    def __init__(self,dimensions,width=16):
        super().__init__();initial=SharedTransformer(dimensions,width)
        self.networks=nn.ModuleDict({c:copy.deepcopy(initial) for c in COMPONENTS})
    def forward(self,c,*x):return self.networks[c](c,*x)


class IndependentPolicy(SharedPolicy):
    def __init__(self,folder):
        self.bundle=read_json(folder/'BUNDLE.json');b=self.bundle
        if b['feature_version']!=FEATURE or b['settings'].get('architecture')!='independent':
            raise ValueError('Wrong independent model')
        verify({str(folder/p):sha for p,sha in b['checkpoint_sha256'].items()})
        self.models=[];self.range_checks={c:{'values':0,'clipped':0} for c in COMPONENTS}
        for p in b['checkpoints']:
            m=IndependentBank(b['dimensions'],b['settings']['width'])
            m.load_state_dict(load_checkpoint(folder/p)['model']);m.eval();self.models.append(m)


def fit_independent(source,out):
    import training
    from data import load_dataset
    b=read_json(source/'shared_model/BUNDLE.json');s=dict(b['settings'],architecture='independent')
    entries=read_json(source/'instances/catalog.json')['instances']
    folders=[source/'collection'/r['instance_id']/arm for r in entries for arm in ('COLLECT_I','COLLECT_CM')]
    for folder in folders:
        if not legacy_runner.completed(folder,read_json(source/'PLAN.json')['sha256']):raise ValueError('Missing source labels')
    rows,norm=load_dataset(folders,s['progress_denominator'])
    if norm!=b['normalization']:raise ValueError('Source normalization could not be reproduced')
    provenance={str(source/'shared_model/BUNDLE.json'):digest(source/'shared_model/BUNDLE.json'),**files()}
    provenance.update({str(f/'DATA.json'):digest(f/'DATA.json') for f in folders})
    oldnet,oldpolicy=training.SharedTransformer,training.SharedPolicy
    training.SharedTransformer=IndependentBank;training.SharedPolicy=IndependentPolicy
    try:result=training.fit(rows,norm,s,out,provenance,dict(b['coverage'],comparison='independent parameters; identical data/loss/scalers/ensemble/selection'))
    finally:training.SharedTransformer=oldnet;training.SharedPolicy=oldpolicy
    for key in ('scalers','normalization','target_scales','training_instances','validation_instances'):
        if result[key]!=b[key]:raise ValueError('Unfair independent control: '+key)
    return result
