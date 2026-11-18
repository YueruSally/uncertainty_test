"""Three task adapters, one shared Transformer backbone; no PPO in this stage."""
from bootstrap import *
from collections import defaultdict
import inspect
import numpy as np
import torch
from torch import nn


def load_checkpoint(path):
    extra = {'weights_only':False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **extra)


def save_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


class SharedTransformer(nn.Module):
    def __init__(self, dimensions, width=16):
        super().__init__()
        self.batch_projection = nn.ModuleDict({c:nn.Linear(dimensions[c][0],width) for c in COMPONENTS})
        self.candidate_projection = nn.ModuleDict({c:nn.Linear(width+dimensions[c][1]+dimensions[c][2]+1,width) for c in COMPONENTS})
        self.task = nn.Embedding(3,width)
        def encoder():
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,
                dropout=0.,activation='gelu',batch_first=True),1)
        self.batch_encoder = encoder(); self.candidate_encoder = encoder()
        self.gain_head = nn.Linear(width,1)

    def forward(self, component, batch, candidate, context, marker):
        b,k,n,f = batch.shape
        h = self.batch_encoder(self.batch_projection[component](batch.reshape(b*k,n,f))).mean(1).reshape(b,k,-1)
        h = self.candidate_projection[component](torch.cat((h,candidate,context[:,None,:].expand(-1,k,-1),marker),-1))
        h = self.candidate_encoder(h+self.task.weight[COMPONENTS.index(component)])
        raw = self.gain_head(h).squeeze(-1)
        return torch.cat((torch.zeros_like(raw[:,:1]),raw[:,1:]),1)


def predictions(models, component, rows, scaler, norm):
    groups = defaultdict(list)
    for i,r in enumerate(rows): groups[len(r['candidate_batch_tokens'][0])].append(i)
    output = np.zeros((len(models),len(rows),8))
    with torch.inference_mode():
        for ids in groups.values():
            for start in range(0,len(ids),32):
                chunk=ids[start:start+32]; x=legacy_learning.pack([rows[i] for i in chunk],scaler,norm)
                for j,m in enumerate(models):
                    m.eval(); output[j,chunk] = m(component,*x).numpy()
    return output


class SharedPolicy:
    def __init__(self, folder):
        self.bundle = read_json(folder / 'BUNDLE.json'); b = self.bundle
        if b['feature_version'] != FEATURE: raise ValueError('Incompatible shared input schema')
        verify({str(folder/p):sha for p,sha in b['checkpoint_sha256'].items()})
        self.models=[]; self.range_checks={c:{'values':0,'clipped':0} for c in COMPONENTS}
        for name in b['checkpoints']:
            m=SharedTransformer(b['dimensions'],b['settings']['width'])
            m.load_state_dict(load_checkpoint(folder/name)['model']);m.eval();self.models.append(m)

    def select(self, row):
        c=row['component'];b=self.bundle;s=b['scalers'][c]
        row=progress_row(row,b['settings']['progress_denominator'])
        for part,x in [('batch',row['candidate_batch_tokens']),('candidate',row['candidate_features']),
                       ('context',legacy_learning.context(row,b['normalization']))]:
            z=np.abs((np.asarray(x)-s[part]['mean'])/s[part]['std'])
            self.range_checks[c]['values']+=int(z.size)
            self.range_checks[c]['clipped']+=int(np.sum(z>s['clip']))
        pred=predictions(self.models,c,[row],s,b['normalization'])[:,0]
        g=b['selected']['gates'][c]
        return legacy_learning.decide(pred,row['legal_mask'],g['lambda'],g['threshold_scaled'],g['disabled'])
