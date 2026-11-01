"""Fixed, training-normalized initialization rule control."""
from __future__ import annotations

from collections import Counter
import math
import random

import baseline_uncertainty as base

from .archive import exact_hv_3d
from .comparison_controllers import BaseBudgetController
from .initialization_rule_features import (
    path_rule_score,
    validate_rule_normalization,
)


DIRECTED_INITIALIZATION_PROFILES = (
    "joint_balanced",
    "joint_balanced_p025",
    "joint_balanced_p050",
    "joint_balanced_p075",
    "joint_balanced_hv_gate",
    "diverse_cycle",
    "time_reliability",
)
INITIALIZATION_RULE_PROFILES = (
    "legacy",
    *DIRECTED_INITIALIZATION_PROFILES,
)
DIVERSE_CYCLE = (
    "joint_balanced",
    "cost",
    "emission",
    "time_reliability",
)
BALANCED_FRACTION_PATTERNS = {
    "joint_balanced_p025": (True, False, False, False),
    "joint_balanced_p050": (True, False),
    "joint_balanced_p075": (True, True, True, False),
}


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


def _finite(value, default=0.0):
    value = float(value)
    return value if math.isfinite(value) else float(default)


def _directed_path_ranking(paths, batch, reliable_options, profile,
                           initial_individual_index):
    """Rank one batch's paths without a real objective evaluation."""
    if profile not in DIRECTED_INITIALIZATION_PROFILES:
        raise ValueError(f"invalid directed I profile: {profile}")
    if profile == "diverse_cycle":
        effective_profile = DIVERSE_CYCLE[
            int(initial_individual_index) % len(DIVERSE_CYCLE)]
    else:
        effective_profile = profile

    option_by_signature = {
        _path_signature(option.path): option
        for option in reliable_options
    }
    finite_lateness = [
        float(option.max_lateness_h)
        for option in reliable_options
        if math.isfinite(float(option.max_lateness_h))
    ]
    missing_lateness = max(finite_lateness, default=0.0) + 1.0
    rows = []
    for path in paths:
        option = option_by_signature.get(_path_signature(path))
        on_time = _finite(
            0.0 if option is None else option.on_time_probability)
        lateness = _finite(
            missing_lateness if option is None else option.max_lateness_h,
            missing_lateness)
        rows.append({
            "path": path,
            "raw": [
                float(batch.quantity) * _finite(path.base_cost_per_teu),
                float(batch.quantity) * _finite(path.base_emission_per_teu),
                _finite(path.base_travel_time_h),
                lateness,
                -on_time,
            ],
            "on_time_probability": on_time,
            "max_lateness_h": lateness,
            "structurally_feasible": option is not None,
        })

    # Scale the actual decision set.  Infeasible fallback rows must not shift
    # the min/max of feasible candidates and thereby change their ordering.
    scaling_rows = [row for row in rows if row["structurally_feasible"]]
    if not scaling_rows:
        scaling_rows = rows
    for column in range(5):
        values = [row["raw"][column] for row in scaling_rows]
        low = min(values)
        span = max(values) - low
        for row in rows:
            row.setdefault("scaled", []).append(
                0.0 if span <= 1e-15
                else (row["raw"][column] - low) / span)

    for row in rows:
        scaled = row["scaled"]
        if effective_profile == "joint_balanced":
            score = (
                scaled[0] + scaled[1]
                + (scaled[2] + scaled[3] + scaled[4]) / 3.0)
        elif effective_profile == "cost":
            score = scaled[0]
        elif effective_profile == "emission":
            score = scaled[1]
        elif effective_profile == "time_reliability":
            score = (scaled[2] + scaled[3] + scaled[4]) / 3.0
        else:
            raise RuntimeError(
                f"unsupported effective I profile: {effective_profile}")
        row["score"] = float(score)
        row["effective_profile"] = effective_profile
    # The fixed teacher must not trade structural feasibility for a low
    # scalar score.  In the normal experiment every batch has many feasible
    # options; retaining the remaining rows only provides a deterministic
    # fallback for malformed/small instances where fewer than k exist.
    rows.sort(key=lambda row: (
        not row["structurally_feasible"],
        row["score"], int(row["path"].path_id),
        _path_signature(row["path"])))
    return rows, effective_profile


class InitializationRuleController(BaseBudgetController):
    """Replace only Base's random initialization path identities by Rule-I."""

    controls_initialization = True
    controls_crossover = False
    controls_mutation = False

    def __init__(self, *args, rule_normalization=None,
                 reliable_options=None, profile="legacy", **kwargs):
        self.profile = str(profile)
        if self.profile not in INITIALIZATION_RULE_PROFILES:
            raise ValueError(
                f"initialization rule profile must be one of "
                f"{INITIALIZATION_RULE_PROFILES}")
        self.rule_normalization = (
            None if rule_normalization is None else dict(rule_normalization))
        if self.profile == "legacy":
            if self.rule_normalization is None:
                raise ValueError("legacy Rule-I requires normalization")
            validate_rule_normalization(self.rule_normalization)
        elif self.rule_normalization is not None:
            raise ValueError(
                "directed Rule-I uses within-batch scaling, not a legacy "
                "normalization")
        self.reliable_options = dict(reliable_options or {})
        if self.profile != "legacy" and not self.reliable_options:
            raise ValueError("directed Rule-I requires reliable path options")
        self.profile_counts = Counter()
        self.hv_gate_enabled = self.profile == "joint_balanced_hv_gate"
        self._gate_pending = None
        self._gate_pairs = []
        self.gate_selected = None
        self.gate_base_hv = None
        self.gate_directed_hv = None
        self.counterfactual_initialization_evaluations = 0
        self.total_attempts = 0
        self.total_effective = 0
        self.no_candidate = 0
        self.rule_choices = 0
        super().__init__(
            *args,
            log_names=("initialization_rule_actions", "initialization_gate",
                       "generation_summary"),
            **kwargs,
        )
        if self.hv_gate_enabled and self.normalizer is None:
            raise ValueError("initialization HV gate requires a normalizer")

    def initial_individual(self, batches, path_lib, max_paths=3):
        individual = base.Individual()
        shadow_individual = base.Individual() if self.hv_gate_enabled else None
        event_id = f"{self.run_id}:RULE-I:{self.total_attempts + 1}"
        self.total_attempts += 1
        initial_individual_index = self.total_attempts - 1
        differs_from_shadow = False
        actions = []
        for batch in batches:
            key = (batch.origin, batch.destination, batch.batch_id)
            paths = list(path_lib.get((batch.origin, batch.destination), ()))
            if not paths:
                self.no_candidate += 1
                continue
            k = random.randint(1, min(int(max_paths), len(paths)))
            shadow_paths = random.sample(paths, k)
            raw_shares = [random.random() for _ in range(k)]
            if shadow_individual is not None:
                shadow_individual.od_allocations[key] = base.merge_and_normalize([
                    base.PathAllocation(path=path, share=raw_shares[index])
                    for index, path in enumerate(shadow_paths)
                ])
            if self.profile == "legacy":
                ranked_rows = [{
                    "path": path,
                    "score": path_rule_score(path, self.rule_normalization),
                    "effective_profile": "legacy",
                    "on_time_probability": None,
                    "max_lateness_h": None,
                    "structurally_feasible": None,
                } for path in sorted(
                    paths,
                    key=lambda path: (
                        path_rule_score(path, self.rule_normalization),
                        int(path.path_id),
                        _path_signature(path),
                    ))]
                effective_profile = "legacy"
            elif (self.profile in BALANCED_FRACTION_PATTERNS
                  and not BALANCED_FRACTION_PATTERNS[self.profile][
                      initial_individual_index
                      % len(BALANCED_FRACTION_PATTERNS[self.profile])]):
                ranked_rows = [{
                    "path": path,
                    "score": None,
                    "effective_profile": "base_passthrough",
                    "on_time_probability": None,
                    "max_lateness_h": None,
                    "structurally_feasible": None,
                } for path in shadow_paths]
                effective_profile = "base_passthrough"
            else:
                ranked_rows, effective_profile = _directed_path_ranking(
                    paths, batch,
                    self.reliable_options.get(key, ()),
                    ("joint_balanced"
                     if (self.profile in BALANCED_FRACTION_PATTERNS
                         or self.hv_gate_enabled)
                     else self.profile),
                    initial_individual_index)
            selected_rows = ranked_rows[:k]
            selected = [row["path"] for row in selected_rows]
            self.profile_counts[effective_profile] += 1
            self.rule_choices += k
            differs_from_shadow = differs_from_shadow or (
                tuple(_path_signature(path) for path in selected)
                != tuple(_path_signature(path) for path in shadow_paths)
            )
            individual.od_allocations[key] = base.merge_and_normalize([
                base.PathAllocation(path=path, share=raw_shares[index])
                for index, path in enumerate(selected)
            ])
            actions.append({
                "batch_id": batch.batch_id,
                "profile": self.profile,
                "effective_profile": effective_profile,
                "required_path_count": k,
                "chosen_path_ids": [path.path_id for path in selected],
                "chosen_scores": [row["score"] for row in selected_rows],
                "chosen_on_time_probabilities": [
                    row["on_time_probability"] for row in selected_rows],
                "chosen_max_lateness_h": [
                    row["max_lateness_h"] for row in selected_rows],
                "chosen_structurally_feasible": [
                    row["structurally_feasible"] for row in selected_rows],
            })
        self.total_effective += int(differs_from_shadow)
        self.append("initialization_rule_actions", {
            "event_id": event_id,
            "profile": self.profile,
            "different_from_shadow_base": bool(differs_from_shadow),
            "actions": actions,
        })
        if self.hv_gate_enabled:
            if self._gate_pending is not None:
                raise RuntimeError("previous initialization pair was not registered")
            self._gate_pending = (individual, shadow_individual)
        return individual

    def initialization_counterfactual(self, individual):
        """Return the paired Base-random individual for fixed-budget gating."""
        if not self.hv_gate_enabled:
            return None
        if self._gate_pending is None:
            return None
        if self._gate_pending[0] is not individual:
            raise RuntimeError("directed initialization does not match gate pair")
        directed, shadow = self._gate_pending
        self._gate_pairs.append((directed, shadow))
        self._gate_pending = None
        self.counterfactual_initialization_evaluations += 1
        return shadow

    def select_initial_population(self, directed_population):
        """Choose the full Base or directed initial population by exact HV."""
        if not self.hv_gate_enabled:
            return directed_population
        if self._gate_pending is not None:
            raise RuntimeError("unregistered initialization gate pair")
        base_by_directed_id = {
            id(directed): shadow for directed, shadow in self._gate_pairs
        }
        base_population = [
            base_by_directed_id.get(id(individual), individual)
            for individual in directed_population
        ]

        def population_hv(population):
            points = [
                self.normalizer.transform(individual.objectives)
                for individual in population if individual.feasible
            ]
            return exact_hv_3d(points, self.normalizer.reference)

        self.gate_directed_hv = population_hv(directed_population)
        self.gate_base_hv = population_hv(base_population)
        use_directed = self.gate_directed_hv >= self.gate_base_hv - 1e-12
        self.gate_selected = "directed" if use_directed else "base"
        selected = directed_population if use_directed else base_population

        # BaseBudgetController observed both alternatives as they were
        # evaluated.  Rebuild its reward-only archive queue from the selected
        # population so search-progress diagnostics never include rejected
        # counterfactual individuals.
        # MutationLogger's canonical fingerprint is needed here; import it
        # lazily to avoid changing baseline_uncertainty's public surface.
        from multi_instance_logging import fingerprint
        self._archive_pending = [
            (individual.objectives, bool(individual.feasible),
             fingerprint(individual))
            for individual in selected
        ]
        self.append("initialization_gate", {
            "base_initial_hv": self.gate_base_hv,
            "directed_initial_hv": self.gate_directed_hv,
            "directed_minus_base_initial_hv": (
                self.gate_directed_hv - self.gate_base_hv),
            "selected": self.gate_selected,
            "shared_population_members": (
                len(directed_population) - len(self._gate_pairs)),
            "paired_random_population_members": len(self._gate_pairs),
            "counterfactual_initialization_evaluations": (
                self.counterfactual_initialization_evaluations),
            "real_objective_evaluation_count": self.evaluations,
        })
        return selected

    def summary(self):
        result = super().summary()
        result.update({
            "initialization_rule_profile": self.profile,
            "initialization_effective_profile_counts": dict(
                self.profile_counts),
            "initialization_attempts": self.total_attempts,
            "initializations_different_from_shadow_base": self.total_effective,
            "no_eligible_candidate": self.no_candidate,
            "rule_choices": self.rule_choices,
            "initialization_hv_gate_enabled": self.hv_gate_enabled,
            "initialization_hv_gate_selected": self.gate_selected,
            "initialization_hv_gate_base_hv": self.gate_base_hv,
            "initialization_hv_gate_directed_hv": self.gate_directed_hv,
            "counterfactual_initialization_evaluation_count": (
                self.counterfactual_initialization_evaluations),
        })
        return result
