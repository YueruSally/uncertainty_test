"""Deterministic linear mutation prior fitted from counterfactual audits."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .config import OPERATORS


def _average_ranks(values):
    result = np.zeros(len(values), dtype=float)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    offset = 0
    while offset < len(ordered):
        end = offset + 1
        while (end < len(ordered)
               and values[ordered[end]] == values[ordered[offset]]):
            end += 1
        rank = (offset + end - 1) / 2.0
        for position in range(offset, end):
            result[ordered[position]] = rank
        offset = end
    return result


def _candidate_absolute_features(observation, indices):
    parent = np.asarray(observation["parent"], dtype=float)
    candidates = np.asarray(observation["candidates"], dtype=float)[indices]
    global_features = np.asarray(observation["global"], dtype=float)
    parent_summary = np.concatenate((parent.mean(axis=0), parent.std(axis=0)))
    operator_id = int(observation["operator_id"])
    operator = np.zeros(len(OPERATORS), dtype=float)
    operator[operator_id] = 1.0
    graph_paths = observation.get("candidate_graph_tokens") or [
        [] for _ in observation["candidates"]]
    rows = []
    for candidate_index, candidate in zip(indices, candidates):
        tokens = graph_paths[candidate_index] or []
        graph = np.asarray(tokens, dtype=float)
        graph_summary = np.asarray([
            float(len(tokens)),
            float(graph[:, 0].mean()) if len(tokens) else 0.0,
            float(graph[:, 2].mean()) if len(tokens) else 0.0,
            float(graph[:, 3].mean()) if len(tokens) else 0.0,
        ])
        rows.append(np.concatenate((
            candidate, global_features, parent_summary,
            graph_summary, operator,
        )))
    return np.stack(rows)


def state_relative_features(absolute):
    standard_deviation = absolute.std(axis=0)
    safe_scale = np.where(standard_deviation > 1e-12,
                          standard_deviation, 1.0)
    relative = (absolute - absolute.mean(axis=0)) / safe_scale
    ranks = np.zeros_like(absolute)
    if len(absolute) > 1:
        for column in range(absolute.shape[1]):
            if standard_deviation[column] <= 1e-12:
                continue
            ranks[:, column] = (
                2.0 * _average_ranks(absolute[:, column].tolist())
                / (len(absolute) - 1) - 1.0)
    return np.concatenate((absolute, relative, ranks), axis=1)


class SupervisedMutationPrior:
    def __init__(self, artifact):
        artifact = dict(artifact or {})
        if artifact.get("schema_version") != 1:
            raise ValueError("unsupported supervised mutation prior schema")
        if artifact.get("feature_mode") != "state_relative":
            raise ValueError("supervised mutation prior requires state_relative features")
        if artifact.get("target_metric", "archive_delta_hv") != "archive_delta_hv":
            raise ValueError("supervised mutation prior must predict archive_delta_hv")
        if artifact.get("selected_operator") != "replace":
            raise ValueError("supervised mutation prior must target replace")
        self.weights = np.asarray(artifact.get("effective_weights"), dtype=float)
        self.alpha = float(artifact.get("alpha"))
        self.temperature = float(artifact.get("temperature"))
        if self.weights.ndim != 1 or not len(self.weights):
            raise ValueError("supervised mutation prior weights must be one-dimensional")
        declared_dimension = artifact.get("feature_dimension")
        if (declared_dimension is not None
                and int(declared_dimension) != len(self.weights)):
            raise ValueError("supervised mutation prior has inconsistent dimensions")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("supervised mutation prior weights must be finite")
        if not 0.0 <= self.alpha <= 1.0 or self.temperature <= 0.0:
            raise ValueError("invalid supervised mutation prior mixture")
        self.artifact = artifact
        blob = json.dumps(
            artifact, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def apply(self, observation):
        eligible = np.asarray(observation["eligible"], dtype=bool)
        indices = np.flatnonzero(eligible)
        if len(indices) < 2 or int(observation["operator_id"]) != 4:
            return observation, None
        absolute = _candidate_absolute_features(observation, indices)
        features = state_relative_features(absolute)
        if features.shape[1] != len(self.weights):
            raise ValueError(
                "supervised mutation prior feature dimension differs from artifact")
        scores = features.dot(self.weights)
        score_ranks = _average_ranks(scores.tolist()) / (len(scores) - 1)
        logits = score_ranks / self.temperature
        logits -= logits.max()
        model = np.exp(logits)
        model /= model.sum()
        base = np.asarray(observation["action_prior"], dtype=float)
        base_legal = base[indices].copy()
        base_mass = float(base_legal.sum())
        if not np.isfinite(base_mass) or base_mass <= 0.0:
            raise ValueError("supervised mutation prior received invalid base prior")
        base_legal /= base_mass
        mixed_legal = (1.0 - self.alpha) * base_legal + self.alpha * model
        mixed = np.zeros_like(base)
        mixed[indices] = mixed_legal
        updated = dict(observation)
        updated["base_action_prior"] = base.tolist()
        updated["action_prior"] = mixed.tolist()
        diagnostics = {
            "digest": self.digest,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "total_variation_from_base": float(
                0.5 * np.abs(mixed - base).sum()),
            "model_top_action": int(indices[int(np.argmax(model))]),
        }
        return updated, diagnostics
