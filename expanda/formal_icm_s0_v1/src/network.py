"""Frozen K8 Transformer architecture and inference feature scaling."""
import numpy as np
import torch
from torch import nn

class CandidateGainTransformer(nn.Module):
    """Predict candidate gain relative to Balanced; Balanced is exactly zero."""
    def __init__(self, batch_feature_count, candidate_feature_count,
                 context_feature_count, width=32):
        super().__init__()
        self.batch_projection = nn.Linear(batch_feature_count, width)
        batch_layer = nn.TransformerEncoderLayer(width, 4, width*2,
            dropout=0., activation='gelu', batch_first=True, norm_first=False)
        self.batch_encoder = nn.TransformerEncoder(batch_layer, 1)
        self.candidate_projection = nn.Linear(
            width+candidate_feature_count+context_feature_count+1, width)
        candidate_layer = nn.TransformerEncoderLayer(width, 4, width*2,
            dropout=0., activation='gelu', batch_first=True, norm_first=False)
        self.candidate_encoder = nn.TransformerEncoder(candidate_layer, 1)
        self.gain_head = nn.Linear(width, 1)

    def forward(self, batch_tokens, candidate_features, context, balanced):
        states, candidates, batches, features = batch_tokens.shape
        hidden = self.batch_projection(batch_tokens.reshape(
            states*candidates, batches, features))
        hidden = self.batch_encoder(hidden).mean(1).reshape(states, candidates, -1)
        repeated = context[:, None, :].expand(-1, candidates, -1)
        hidden = self.candidate_projection(torch.cat(
            (hidden, candidate_features, repeated, balanced), dim=-1))
        raw = self.gain_head(self.candidate_encoder(hidden)).squeeze(-1)
        return torch.cat((torch.zeros_like(raw[:, :1]), raw[:, 1:]), dim=1)


def state_context(row):
    points = np.asarray(row['population_points'], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('state lacks a finite three-objective population')
    context = np.concatenate((points.mean(0), points.std(0),
        points.min(0), points.max(0),
        [float(row['generation'])/20., float(row['evaluation'])/400.]))
    if not np.isfinite(context).all():
        raise ValueError('non-finite prescreen context')
    return context


def scaled(values, part, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler[part]['mean'], dtype=np.float32)
    std = np.asarray(scaler[part]['std'], dtype=np.float32)
    return np.clip((values-mean)/std, -scaler['clip'], scaler['clip'])
