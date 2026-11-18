from copy import deepcopy
from transformer_ppo.observations import joint_balanced_candidate_scores
import time
import numpy as np
import baseline_uncertainty as base
from m_candidate_information import (CandidateController, build_proposals, private_seed, proposal_features, proposal_randomness)
from m_operator_fixed_mix import FixedMixPolicy
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
POOL_SIZE = 8

def batch_tokens(child, batches):
    """All batch allocations are represented before any objective evaluation."""
    from m_candidate_information import allocation_metrics
    total_quantity = max(sum(float(b.quantity) for b in batches), 1e-8)
    tokens = []
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        features = allocation_metrics(child.od_allocations.get(key, []), batch.quantity)
        features = np.concatenate((
            [float(batch.quantity)/total_quantity,
             float(batch.ET)/1000., float(batch.LT)/1000.], features))
        if not np.isfinite(features).all():
            raise ValueError('non-finite candidate batch token')
        tokens.append(features.tolist())
    return tokens


def build_replace_pool(controller, parent, batches, path_lib, tt_dict,
                       arc_lookup, reliable_options):
    """Exact Balanced child plus up to seven distinct legal Replace children."""
    operator_candidates = controller._operator_candidates(
        parent, batches, path_lib, tt_dict, arc_lookup, reliable_options)
    balanced = build_proposals(
        controller, parent, batches, path_lib, tt_dict, arc_lookup,
        reliable_options, operator_candidates)[4]
    balanced = {**balanced, 'eligible': bool(operator_candidates[4]['eligible']
                and balanced['ok']), 'source': 'balanced'}
    pool = [balanced]
    seen = {balanced['fingerprint']}
    if balanced['eligible']:
        rows = controller._operator_target_rows(parent, batches, 'replace',
            path_lib, tt_dict, arc_lookup, reliable_options)
        eligible = [bool(row.get('eligible')) for row in rows]
        scores = joint_balanced_candidate_scores(rows, eligible)
        order = sorted((i for i, legal in enumerate(eligible) if legal),
                       key=lambda i: (float(scores[i]),
                           int(rows[i]['target'].batch_id),
                           str(rows[i].get('path_id'))))
        batch_by_id = {b.batch_id: b for b in batches}
        for index in order:
            if len(pool) == POOL_SIZE:
                break
            row = rows[index]
            batch = batch_by_id[row['target'].batch_id]
            child = deepcopy(parent)
            seed = private_seed('prescreen-k8-v1', controller.run_id,
                controller.seeds['algorithm'], controller.generation,
                controller._active_mixture_event_index, index)
            with proposal_randomness(seed):
                ok = bool(base.apply_mutation_op(child, 'replace', batch,
                    path_lib, tt_dict, arc_lookup,
                    reliable_options=reliable_options, target=row['target']))
                if ok:
                    base.repair_missing_allocations(child, batches, path_lib)
            identity = fingerprint(child)
            if not ok or identity in seen:
                continue
            seen.add(identity)
            pool.append({'operator': 'replace', 'child': child, 'batch': batch,
                'target': row['target'], 'ok': ok, 'metadata': {'selector': 'prescreen-k8'},
                'seed': seed, 'fingerprint': identity, 'eligible': True,
                'source': 'alternative',
                'features': proposal_features(parent, child, batch, row['target'],
                    batches, ok)})
    return pool


class CollectBalancedPolicy(FixedMixPolicy):
    """The collector searches with Balanced; labels never alter decisions."""
    def __init__(self, instance, seed):
        super().__init__(instance, seed, 0.)
        self.name = 'prescreen-k8-collect-balanced'
        self.collect = True

    def choose(self, candidates, context=None, **kwargs):
        if not candidates or len(candidates) > POOL_SIZE:
            raise ValueError('invalid Replace pool size')
        chosen = 0 if candidates[0]['eligible'] else None
        probabilities = [float(i == chosen) for i in range(len(candidates))]
        return Decision(chosen, probabilities, [None]*len(candidates),
            {'decision_kind': 'prescreen_collect_balanced', 'inference_seconds': 0.})


class CollectK8Controller(CandidateController):

    """Evaluate alternatives only for saved training labels, outside B400."""
    def _execute_operator_action(self, ind, operator, batches, path_lib,
                                 tt_dict, arc_lookup, reliable_options,
                                 target_rng=None, trace=None):
        if operator != 'replace' or fingerprint(ind) != self.proposal_parent:
            raise RuntimeError('Replace pool execution contract changed')
        proposal = self.proposals[self.selected_proposal]
        ind.od_allocations = deepcopy(proposal['child'].od_allocations)
        if fingerprint(ind) != proposal['fingerprint']:
            raise RuntimeError('selected K8 child was not executed exactly')
        self.execution_checks += 1
        return (proposal['ok'], proposal['batch'], proposal['target'],
            {**proposal['metadata'], 'proposal_seed': proposal['seed'],
             'proposal_exact': True})
    def summary(self):
        return {**super().summary(), 'candidate_contract': 'prescreen-k8-v1'}

"""Frozen size-200 K8 ranker used by the bounded development search."""

from collections import Counter
import time

import numpy as np
import torch

from network import (CandidateGainTransformer, scaled,
                                   state_context)
from multi_instance_logging import fingerprint
from multi_instance_policy import Decision
from transformer_ppo.controller import PPOController


METHODS = ("HEURISTIC_BALANCED", "TRANSFORMER_K8_SIZE200")
PPO_METHOD_PREFIX = "TRANSFORMER_K8_PPO_U"


def is_transformer_method(method):
    return method == "TRANSFORMER_K8_SIZE200" or method.startswith(
        PPO_METHOD_PREFIX)


def choose_first_maximum(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("ranker predictions must be a finite vector")
    best = float(values.max())
    return next(index for index, value in enumerate(values)
                if value >= best-1e-12)


def pack_inference_row(row, scaler):
    batch = np.stack([scaled(
        row["candidate_batch_tokens"], "batch", scaler)])
    candidate = np.stack([scaled(
        row["candidate_features"], "candidate", scaler)])
    context = scaled(np.stack([state_context(row)]), "context", scaler)
    balanced = np.zeros((1, candidate.shape[1], 1), np.float32)
    balanced[:, 0, 0] = 1.0
    return tuple(torch.as_tensor(value) for value in (
        batch, candidate, context, balanced))


class FrozenScaleRanker:
    """Load the fixed size-200 three-seed ensemble once per search run."""
    def __init__(self, model_bundle, digest_function):
        selected = model_bundle.get("models", {}).get("200")
        if selected is None:
            raise ValueError("scale bundle has no size-200 model")
        self.scaler = selected["scaler"]
        self.target_scale = float(selected["target_scale"])
        self.models = []
        for filename in selected["checkpoints"]:
            expected = selected["checkpoint_sha256"].get(filename)
            if expected is None or digest_function(filename) != expected:
                raise ValueError(f"ranker checkpoint digest mismatch: {filename}")
            checkpoint = torch.load(filename, map_location="cpu")
            if (checkpoint.get("protocol") not in (
                    "m-prescreen-scale-ranker-v1", "m-prescreen-scale-ppo-v1")
                    or checkpoint.get("training_instances") != 200
                    or checkpoint.get("width") != 16):
                raise ValueError("ranker checkpoint identity mismatch")
            model = CandidateGainTransformer(
                checkpoint["batch_feature_count"],
                checkpoint["candidate_feature_count"],
                checkpoint["context_feature_count"],
                width=checkpoint["width"])
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("size-200 ranker must contain three models")

    def predict(self, row):
        packed = pack_inference_row(row, self.scaler)
        with torch.inference_mode():
            predictions = np.mean([
                model(*packed).numpy()*self.target_scale
                for model in self.models], axis=0)[0]
        if predictions.shape != (8,) or abs(float(predictions[0])) > 1e-12:
            raise ValueError("size-200 ranker output contract changed")
        return predictions


class ScaleSearchPolicy(CollectBalancedPolicy):
    def __init__(self, instance, seed, method, predictor=None):
        if method != "HEURISTIC_BALANCED" and not is_transformer_method(method):
            raise ValueError(f"unknown scale-search method: {method}")
        if is_transformer_method(method) != (predictor is not None):
            raise ValueError("predictor must be supplied only for Transformer")
        super().__init__(instance, seed)
        self.search_method = method
        self.predictor = predictor
        self.name = f"prescreen-scale-search-{method.lower()}"


class ScaleSearchController(CollectK8Controller):
    """Select one unevaluated K8 child; never collect counterfactual labels."""
    def __init__(self, *args, **kwargs):
        trainer = kwargs["trainer"]
        self.search_method = trainer.search_method
        self.predictor = trainer.predictor
        self.pool_events = 0
        self.unsupported_events = 0
        self.pool_candidates = 0
        self.selection_counts = Counter()
        self.ranker_inference_seconds = 0.0
        super().__init__(*args, **kwargs)

    def _mutate_operator_only(
            self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options,
            mixture_event_index, evaluation_kwargs):
        # Bypass CollectK8Controller's training-label evaluations.  Both
        # methods evaluate only the one child charged to the common B400.
        return PPOController._mutate_operator_only(
            self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options,
            mixture_event_index, evaluation_kwargs)

    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup,
                         reliable_options):
        started = time.perf_counter()
        self.proposal_parent = fingerprint(ind)
        self.proposals = build_replace_pool(
            self, ind, batches, path_lib, tt_dict, arc_lookup,
            reliable_options)
        if not self.proposals or not self.proposals[0]["eligible"]:
            self.unsupported_events += 1
            self.selected_proposal = None
            return Decision(None, [0.0], [0.0], {
                "decision_kind": "k8_scale_search_unsupported",
                "search_method": self.search_method,
                "pool_size": 0, "switched": False,
                "inference_seconds": 0.0}), [
                    {"candidate_operator": "replace", "eligible": False}]
        self.pool_events += 1
        self.pool_candidates += len(self.proposals)
        predictions = np.zeros(len(self.proposals), dtype=float)
        points = [
            list(self.archive.normalizer.transform(member.objectives))
            for member in self.current_population if member.feasible]
        inference_seconds = 0.0
        if is_transformer_method(self.search_method) \
                and len(self.proposals) == 8 and points:
            row = {
                "generation": self.generation,
                "evaluation": self.evaluations,
                "population_points": points,
                "candidate_features": [
                    proposal["features"].tolist()
                    for proposal in self.proposals],
                "candidate_batch_tokens": [
                    batch_tokens(proposal["child"], batches)
                    for proposal in self.proposals],
            }
            inference_started = time.perf_counter()
            predictions = self.predictor.predict(row)
            inference_seconds = time.perf_counter()-inference_started
            self.ranker_inference_seconds += inference_seconds
            selected = choose_first_maximum(predictions)
        else:
            # Balanced is also the mandatory fallback for a non-exact pool.
            selected = 0
            if is_transformer_method(self.search_method):
                self.unsupported_events += 1
        self.selected_proposal = selected
        self.selection_counts[str(selected)] += 1
        self.proposal_seconds += time.perf_counter()-started
        candidates = [{"candidate_operator": "replace",
            "eligible": bool(row["eligible"])} for row in self.proposals]
        probabilities = [float(index == selected)
                         for index in range(len(self.proposals))]
        return Decision(selected, probabilities, predictions.tolist(), {
            "decision_kind": "k8_scale_search",
            "search_method": self.search_method,
            "pool_size": len(self.proposals),
            "switched": selected != 0,
            "inference_seconds": inference_seconds,
        }), candidates

    def summary(self):
        selections = sum(self.selection_counts.values())
        return {**super().summary(), "k8_scale_search": {
            "method": self.search_method,
            "pool_events": self.pool_events,
            "unsupported_events": self.unsupported_events,
            "pool_candidates": self.pool_candidates,
            "selection_counts": dict(self.selection_counts),
            "switches": selections-self.selection_counts.get("0", 0),
            "ranker_inference_seconds": self.ranker_inference_seconds,
            "hidden_candidate_objective_evaluations": 0,
        }}
