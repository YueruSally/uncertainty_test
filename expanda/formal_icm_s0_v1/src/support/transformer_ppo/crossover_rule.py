"""Deterministic directed-balanced teacher for common-node crossover."""
from __future__ import annotations

import math

import numpy as np

from .comparison_controllers import BaseBudgetController
from .crossover import apply_cut, finalize_children, prepare_common_node_crossover


CROSSOVER_RULE_PROFILES = (
    "joint_balanced",
    "joint_balanced_safe_all",
    "joint_balanced_single",
    "joint_balanced_safe_single",
)


def _finite(value, default=0.0):
    value = float(value)
    return value if math.isfinite(value) else float(default)


def _candidate_proxy(candidate):
    """Return the five minimisation proxies used by directed C."""
    candidate = dict(candidate)
    mode_cv = {"road": 0.15, "rail": 0.10, "water": 0.20}
    deadline = max(1.0, _finite(candidate.get("deadline_window_h"), 1.0))
    child_time = _finite(candidate.get("child_combined_time"))
    modal_cv = 0.5 * sum(
        mode_cv[mode] * (
            _finite(candidate.get(f"child_1_{mode}_fraction"))
            + _finite(candidate.get(f"child_2_{mode}_fraction")))
        for mode in mode_cv)
    deadline_pressure = max(0.0, child_time - deadline) / deadline
    uncertainty_exposure = (
        child_time * modal_cv / deadline
        + 0.5 * _finite(candidate.get("child_border_fraction")))
    return [
        _finite(candidate.get("child_cost_delta")),
        _finite(candidate.get("child_emission_delta")),
        _finite(candidate.get("child_time_delta")),
        deadline_pressure,
        uncertainty_exposure,
    ]


def _scaled_balanced_rows(rows):
    """Scale one candidate pool once and attach the balanced score."""
    if not rows:
        return rows
    values = np.asarray([row["raw"] for row in rows], dtype=float)
    minima = values.min(axis=0)
    spans = values.max(axis=0) - minima
    spans[spans <= 1e-15] = 1.0
    scaled = (values - minima) / spans
    for row, vector in zip(rows, scaled):
        row["scaled"] = vector.tolist()
        row["score"] = float(
            vector[0] + vector[1]
            + (vector[2] + vector[3] + vector[4]) / 3.0)
    return rows


def _directed_cut_ranking(candidates, profile="joint_balanced"):
    """Rank legal cut nodes without any real objective evaluation.

    New crossover paths are not members of the precomputed reliable-path
    library, so the final two risk terms are deterministic zero-cost proxies:
    deadline pressure and modal/border uncertainty exposure.
    """
    if profile not in CROSSOVER_RULE_PROFILES:
        raise ValueError(f"invalid directed C profile: {profile}")
    rows = []
    for index, source in enumerate(candidates):
        candidate = dict(source)
        raw = _candidate_proxy(candidate)
        rows.append({
            "candidate": candidate,
            "candidate_index": index,
            "raw": raw,
            "deadline_pressure": raw[3],
            "uncertainty_exposure": raw[4],
        })
    _scaled_balanced_rows(rows)
    rows.sort(key=lambda row: (
        row["score"],
        int(row["candidate_index"]),
        str(row["candidate"].get("join_node"))))
    return rows


def _safe_balanced_override(ranked, shadow_base_index, tolerance=1e-12):
    """Return the best candidate that Pareto-dominates the paired Base cut."""
    shadow = next(
        row for row in ranked
        if row["candidate_index"] == int(shadow_base_index))
    safe = []
    for row in ranked:
        if row["candidate_index"] == int(shadow_base_index):
            continue
        weakly_better = all(
            candidate <= reference + tolerance
            for candidate, reference in zip(row["raw"], shadow["raw"]))
        strictly_better = any(
            candidate < reference - tolerance
            for candidate, reference in zip(row["raw"], shadow["raw"]))
        if weakly_better and strictly_better:
            safe.append(row)
    return (safe[0] if safe else shadow), shadow


def _joint_single_override_plans(decisions, safe=False, tolerance=1e-12):
    """Choose at most one Batch x cut action from one event-wide pool.

    Every action is expressed relative to the paired Base shadow action for
    its Batch.  The no-override Base action is included in the single shared
    scaling pool, so unlike the legacy all-cut rule this is one joint choice.
    """
    plans = [{
        "cut_index": cut_index,
        "candidate_index": int(cut.shadow_base_index),
        "metadata": None,
        "score_improvement": 0.0,
    } for cut_index, cut in enumerate(decisions)]
    actions = [{
        "is_base": True,
        "cut_index": -1,
        "candidate_index": -1,
        "raw": [0.0] * 5,
        "candidate": None,
    }]
    for cut_index, cut in enumerate(decisions):
        if len(cut.candidates) < 2:
            continue
        shadow_raw = _candidate_proxy(
            cut.candidates[int(cut.shadow_base_index)])
        for candidate_index, candidate in enumerate(cut.candidates):
            if candidate_index == int(cut.shadow_base_index):
                continue
            candidate_raw = _candidate_proxy(candidate)
            actions.append({
                "is_base": False,
                "cut_index": cut_index,
                "candidate_index": candidate_index,
                "raw": [
                    candidate_value - shadow_value
                    for candidate_value, shadow_value in zip(
                        candidate_raw, shadow_raw)
                ],
                "candidate_raw": candidate_raw,
                "shadow_raw": shadow_raw,
                "candidate": dict(candidate),
            })
    _scaled_balanced_rows(actions)
    base_action = actions[0]
    alternatives = actions[1:]
    if safe:
        alternatives = [
            action for action in alternatives
            if all(value <= tolerance for value in action["raw"])
            and any(value < -tolerance for value in action["raw"])
        ]
    if not alternatives:
        return plans, base_action
    chosen = min(alternatives, key=lambda action: (
        action["score"], action["cut_index"], action["candidate_index"]))
    plan = plans[int(chosen["cut_index"])]
    plan["candidate_index"] = int(chosen["candidate_index"])
    plan["metadata"] = chosen
    plan["score_improvement"] = float(
        base_action["score"] - chosen["score"])
    return plans, chosen


class CrossoverRuleController(BaseBudgetController):
    """Apply legacy all-cut or event-wide joint-single crossover rules."""

    controls_crossover = True

    def __init__(self, *args, profile="joint_balanced", min_generation=0,
                 max_generation=None, **kwargs):
        if profile not in CROSSOVER_RULE_PROFILES:
            raise ValueError(f"invalid directed C profile: {profile}")
        self.profile = str(profile)
        self.min_generation = int(min_generation)
        if self.min_generation < 0:
            raise ValueError("directed C minimum generation cannot be negative")
        self.max_generation = (
            None if max_generation is None else int(max_generation))
        if (self.max_generation is not None
                and self.max_generation < self.min_generation):
            raise ValueError(
                "directed C maximum generation cannot precede its minimum")
        self.directed_events = 0
        self.directed_cut_count = 0
        self.overridden_cut_count = 0
        self.single_override_violations = 0
        self._active_crossover = None
        kwargs["log_names"] = (
            "crossover_rule_events", "generation_summary")
        super().__init__(*args, **kwargs)

    def crossover_common_node(
            self, parent1, parent2, batches, tt_dict, arc_lookup):
        child1, child2, decisions = prepare_common_node_crossover(
            parent1, parent2, batches, tt_dict=tt_dict)
        self.total_attempts += 1
        directed = (
            self.generation >= self.min_generation
            and (self.max_generation is None
                 or self.generation <= self.max_generation))
        self.directed_events += int(directed and bool(decisions))
        eligible_cut_count = sum(
            len(cut.candidates) >= 2 for cut in decisions)
        self.directed_cut_count += int(directed) * eligible_cut_count
        single_profiles = (
            "joint_balanced_single", "joint_balanced_safe_single")
        if directed and self.profile in single_profiles:
            plans, _ = _joint_single_override_plans(
                decisions,
                safe=self.profile == "joint_balanced_safe_single")
        else:
            plans = []
            for cut_index, cut in enumerate(decisions):
                if not directed or len(cut.candidates) < 2:
                    plans.append({
                        "cut_index": cut_index,
                        "candidate_index": int(cut.shadow_base_index),
                        "metadata": None,
                        "score_improvement": 0.0,
                    })
                    continue
                ranked = _directed_cut_ranking(cut.candidates, self.profile)
                if self.profile == "joint_balanced":
                    chosen = ranked[0]
                    shadow = next(
                        row for row in ranked
                        if row["candidate_index"]
                        == int(cut.shadow_base_index))
                else:
                    chosen, shadow = _safe_balanced_override(
                        ranked, cut.shadow_base_index)
                plans.append({
                    "cut_index": cut_index,
                    "candidate_index": int(chosen["candidate_index"]),
                    "metadata": chosen,
                    "score_improvement": float(
                        shadow["score"] - chosen["score"]),
                })

        selected = []
        changed = False
        event_override_count = 0
        for plan, cut in zip(plans, decisions):
            cut_index = int(plan["cut_index"])
            candidate_index = int(plan["candidate_index"])
            metadata = plan["metadata"]
            candidate = cut.candidates[candidate_index]
            changed = apply_cut(
                child1, child2, cut, candidate, tt_dict, arc_lookup) or changed
            overridden = candidate_index != int(cut.shadow_base_index)
            self.overridden_cut_count += int(directed and overridden)
            event_override_count += int(directed and overridden)
            selected.append({
                "cut_index": cut_index,
                "batch_id": int(cut.batch.batch_id),
                "candidate_count": len(cut.candidates),
                "shadow_base_index": int(cut.shadow_base_index),
                "chosen_index": candidate_index,
                "join_node": candidate.get("join_node"),
                "different_from_shadow_base": overridden,
                "score": None if metadata is None else metadata["score"],
                "score_improvement_over_shadow_base": (
                    float(plan["score_improvement"])),
                "raw_proxy_vector": (
                    None if metadata is None else metadata["raw"]),
                "scaled_proxy_vector": (
                    None if metadata is None else metadata["scaled"]),
            })
        if self.profile in single_profiles and event_override_count > 1:
            self.single_override_violations += 1
        finalize_children(child1, child2)
        self.total_effective += int(changed)
        self._active_crossover = {
            "generation": int(self.generation),
            "directed": bool(directed),
            "decision_count": len(decisions),
            "changed": bool(changed),
            "selected_cuts": selected,
        }
        if not decisions:
            self.no_candidate += 1
        return child1, child2

    def settle_crossover(self, child1, child2):
        event = self._active_crossover
        if event is None:
            # Structural crossover does not pass through this controller.
            return
        self.append("crossover_rule_events", {
            **event,
            "child_objectives": [
                list(map(float, child1.objectives)),
                list(map(float, child2.objectives)),
            ],
            "child_feasible": [bool(child1.feasible), bool(child2.feasible)],
        })
        self._active_crossover = None

    def summary(self):
        return {
            **super().summary(),
            "crossover_rule_profile": self.profile,
            "crossover_rule_min_generation": self.min_generation,
            "crossover_rule_max_generation": self.max_generation,
            "directed_crossover_events": self.directed_events,
            "directed_cut_count": self.directed_cut_count,
            "overridden_cut_count": self.overridden_cut_count,
            "single_override_violations": self.single_override_violations,
        }
