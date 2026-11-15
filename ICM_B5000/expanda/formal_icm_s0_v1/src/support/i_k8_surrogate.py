import numpy as np
import torch
from network import CandidateGainTransformer

def state_context(row):
    points = np.asarray(row["population_points"], dtype=np.float64)
    if points.size == 0:
        summary = np.zeros(12, dtype=np.float64)
    else:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("I-K8 population points must be Nx3")
        summary = np.concatenate((
            points.mean(0), points.std(0), points.min(0), points.max(0)))
    context = np.concatenate((summary, [
        float(row["slot_fraction"]),
        len(points) / 20.0,
    ]))
    if not np.isfinite(context).all():
        raise ValueError("non-finite I-K8 context")
    return context


def _scaled(values, part, scaler):
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(scaler[part]["mean"], dtype=np.float32)
    standard = np.asarray(scaler[part]["std"], dtype=np.float32)
    return np.clip((values - mean) / standard,
                   -float(scaler["clip"]), float(scaler["clip"]))


def pack_rows(rows, scaler):
    batch = np.stack([
        _scaled(row["candidate_batch_tokens"], "batch", scaler)
        for row in rows
    ])
    candidate = np.stack([
        _scaled(row["candidate_features"], "candidate", scaler)
        for row in rows
    ])
    context = _scaled(
        np.stack([state_context(row) for row in rows]), "context", scaler)
    base = np.zeros(
        (len(rows), len(rows[0]["candidate_features"]), 1), np.float32)
    base[:, 0, 0] = 1.0
    return tuple(torch.as_tensor(value) for value in (
        batch, candidate, context, base))
