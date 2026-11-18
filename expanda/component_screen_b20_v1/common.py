"""Shared, versioned IO and immutable baseline imports for B20 component screening."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
ENGINE=ROOT.parent/'formal_icm_s0_v1/src'
sys.path.insert(0,str(ENGINE/'support'));sys.path.insert(0,str(ENGINE))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLBACKEND']='Agg'
from runtime_common import read_json,write_json,digest
from transformer_ppo.instance_catalog import canonical_digest,read_instance,SCENARIOS
METHODS=('BASE','I_RULE','I_LEARN','C_RULE','C_LEARN','M_RULE','M_LEARN')
FEATURE_VERSION='b20-residual-v1-evaluation-progress-population-fraction'
def sources():
    return sorted(list(ROOT.glob('*.py'))+list(ENGINE.rglob('*.py'))+[ENGINE/'data/border_crossing_events.csv'])
def freeze(path,value):
    if path.exists() and read_json(path)!=value:raise ValueError(f'Frozen inputs changed: {path}. Preserve this output and use a new version.')
    write_json(path,value)
def divisor(instance):
    bs=instance['batches'];q=max(1.,sum(b['quantity'] for b in bs))
    return [q,q,max(1.,max(b['LT'] for b in bs)-min(b['ET'] for b in bs))]
def safe_objectives(ind):
    import math
    vals=[float(v) for v in ind.objectives]
    if not all(math.isfinite(v) for v in vals):
        if ind.feasible:raise ValueError('Feasible nonfinite objective')
        return None
    return vals
