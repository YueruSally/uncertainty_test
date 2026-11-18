from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import random
import time
import numpy as np
import baseline_uncertainty as base
from m_operator_fixed_mix import FixedMixController
from m_operator_closed_loop import FrozenOperatorController
from multi_instance_logging import fingerprint
OPERATORS = ("add", "del", "mod", "mode", "replace")

def private_seed(*parts):
    value = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


@contextmanager
def proposal_randomness(seed):
    """Local construction cannot consume the real search's random stream."""
    py, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def allocation_metrics(allocations, quantity):
    v = np.zeros(len(METRICS))
    v[5] = len(allocations)
    for allocation in allocations:
        p, w = allocation.path, float(allocation.share)
        n = max(1, len(p.arcs))
        v[:5] += w * np.array([p.base_cost_per_teu, p.base_emission_per_teu,
            p.base_travel_time_h, sum(a.distance for a in p.arcs),
            min((a.capacity for a in p.arcs), default=0.) / max(quantity, 1e-8)])
        v[6] += w * len(p.arcs)
        v[7:10] += w * np.array([sum(a.mode == mode for a in p.arcs) / n for mode in ("road", "rail", "water")])
        v[10] += w * sum(bool(a.is_border_arc) for a in p.arcs) / n
    return v


def proposal_features(parent, child, batch, target, batches, ok):
    """No objectives, feasibility, scenario samples, rank or HV may enter here."""
    if batch is None:
        return np.zeros(len(SPECIFIC_NAMES))
    key = (batch.origin, batch.destination, batch.batch_id)
    old = parent.od_allocations.get(key, [])
    before = allocation_metrics(old, batch.quantity)
    after = allocation_metrics(child.od_allocations.get(key, []), batch.quantity)
    # Same pre-action denominator for both before/after. No CCP evaluation.
    population_scale = np.mean([allocation_metrics(parent.od_allocations.get(
        (b.origin, b.destination, b.batch_id), []), b.quantity) for b in batches], axis=0)
    scale = np.maximum(np.abs(population_scale), 1.)
    index = None if target is None else target.allocation_index
    arc_index = None if target is None else target.arc_index
    arc_count = len(old[index].path.arcs) if index is not None and index < len(old) else 0
    prefix = [next(i for i, b in enumerate(batches) if b.batch_id == batch.batch_id) / max(1, len(batches)-1),
        batch.quantity / max(sum(b.quantity for b in batches), 1e-8),
        max(0., batch.LT-batch.ET) / max(1., before[2]),
        -1. if index is None else index / max(1, len(old)-1),
        -1. if arc_index is None else arc_index / max(1, arc_count-1),
        float(ok), float(fingerprint(parent) != fingerprint(child))]
    value = np.concatenate([prefix, before/scale, after/scale, (after-before)/scale])
    if value.shape != (len(SPECIFIC_NAMES),) or not np.isfinite(value).all():
        raise ValueError("invalid pre-evaluation candidate features")
    return value


def build_proposals(controller, parent, batches, path_lib, tt_dict, arc_lookup, reliable_options, candidates):
    """The same proposals serve both information arms and Balanced."""
    result = []
    for action, op in enumerate(OPERATORS):
        child, batch, target, ok, metadata = deepcopy(parent), None, None, False, {}
        seed = private_seed("candidate-info-v1", controller.run_id, controller.seeds["algorithm"],
                            controller.generation, controller._active_mixture_event_index, op)
        if candidates[action]["eligible"]:
            with proposal_randomness(seed):
                # Explicit base-class call prevents recursively selecting an already frozen child.
                ok, batch, target, metadata = FixedMixController._execute_operator_action(
                    controller, child, op, batches, path_lib, tt_dict, arc_lookup, reliable_options,
                    target_rng=random.Random(private_seed(seed, "target")))
        result.append({"operator": op, "child": child, "batch": batch, "target": target,
            "ok": ok, "metadata": metadata, "seed": seed, "fingerprint": fingerprint(child),
            "features": proposal_features(parent, child, batch, target, batches, ok)})
    return result


class CandidateController(FixedMixController):

    def __init__(self, *args, **kwargs):
        self.feedback_rows, self.feedback_evaluations, self.execution_checks = [], 0, 0
        self.proposal_seconds = 0.
        self.proposals, self.proposal_parent = None, None
        super().__init__(*args, **kwargs)
    def summary(self):
        return {**FrozenOperatorController.summary(self), "candidate_contract": "concrete-proposal-v1",
            "successful_actions": dict(self.successful_actions), "effective_actions": dict(self.effective_actions),
            "candidate_construction_seconds": self.proposal_seconds,
            "selected_candidate_execution_checks": self.execution_checks,
            "training_feedback_evaluations": self.feedback_evaluations,
            "training_feedback_states": len(self.feedback_rows)}


METRICS = ("cost", "emission", "nominal_time", "distance", "capacity_per_quantity",
           "path_count", "weighted_arc_count", "road_share", "rail_share", "water_share", "border_share")
SPECIFIC_NAMES = ("batch_position", "quantity_fraction", "deadline_window_relative",
                  "allocation_position", "arc_position", "constructor_success", "encoding_changed") + tuple(
    f"{prefix}_{name}" for prefix in ("before", "after", "delta") for name in METRICS)
