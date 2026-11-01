"""PPO controller implementing the frozen NSGA-II mutation logger hook."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import random
import time
import math

import numpy as np

import baseline_uncertainty as base
from learning_mutation import repair_after_mutation
from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_mutation import candidates_for, replacement_path_candidates

from .archive import RewardArchive, RewardConfig, exact_hv_3d, hv_reward
from .config import OPERATORS
from .observations import build_observation
from .v2_controls import private_seed, validate_model_choice_probability


class PPOController(MutationLogger):
    """Collect action rewards without changing environmental selection."""

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size,
                 graph_observation=None, model_choice_probability=1.0,
                 model_choice_event=None,
                 forced_candidate_index=None,
                 forced_operator=None,
                 forced_operator_min_generation=None,
                 forced_operator_probability=1.0,
                 forced_operator_positive_probe=False,
                 forced_operator_reversible_probe=False,
                 forced_operator_probe_max_attempts=1,
                 forced_operator_probe_retry_events=0,
                 forced_operator_best_of=1,
                 forced_replace_selector="base_random",
                 continuation_event=None, continuation_seed=None,
                 candidate_audit_states=0, candidate_audit_candidates=4,
                 candidate_audit_replicates=1,
                 **kwargs):
        self.trainer = trainer
        self._ppo_updates_at_start = int(trainer.updates)
        self.archive = archive
        self.reward_config = reward_config
        self.batches_for_observation = batches
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.graph_observation = graph_observation
        self.current_population = []
        self.pending_offspring = []
        self._pending_fingerprints = set()
        self.current_individual = None
        self._suppress_archive = False
        self._inside_mutation = False
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        self.model_control_events = 0
        self.base_control_events = 0
        self.base_passthrough_events = 0
        self.model_choice_probability = validate_model_choice_probability(
            model_choice_probability)
        self.model_choice_event = (
            None if model_choice_event is None else int(model_choice_event))
        if self.model_choice_event is not None and self.model_choice_event < 1:
            raise ValueError("model_choice_event must be a positive integer")
        if (self.model_choice_event is not None
                and self.model_choice_probability != 1.0):
            raise ValueError(
                "model_choice_event cannot be combined with probability gating")
        self.forced_candidate_index = (
            None if forced_candidate_index is None
            else int(forced_candidate_index))
        if self.forced_candidate_index is not None:
            if self.forced_candidate_index < 0:
                raise ValueError("forced M candidate index cannot be negative")
            if self.model_choice_event is None:
                raise ValueError(
                    "forced M candidate index requires model_choice_event")
        self.forced_operator = (
            None if forced_operator is None else str(forced_operator))
        if self.forced_operator is not None:
            if self.forced_operator not in OPERATORS:
                raise ValueError(
                    f"forced M operator must be one of {tuple(OPERATORS)}")
            if self.forced_candidate_index is not None:
                raise ValueError(
                    "forced M operator and candidate index are mutually exclusive")
        self.forced_operator_min_generation = (
            None if forced_operator_min_generation is None
            else int(forced_operator_min_generation))
        self.forced_operator_probability = validate_model_choice_probability(
            forced_operator_probability)
        self.forced_operator_positive_probe = bool(
            forced_operator_positive_probe)
        self.forced_operator_reversible_probe = bool(
            forced_operator_reversible_probe)
        self.forced_operator_probe_max_attempts = int(
            forced_operator_probe_max_attempts)
        self.forced_operator_probe_retry_events = int(
            forced_operator_probe_retry_events)
        self.forced_operator_best_of = int(forced_operator_best_of)
        self.forced_replace_selector = str(forced_replace_selector)
        replace_selectors = (
            "base_random", "joint_extreme_cycle", "joint_balanced",
            "joint_pareto_cycle", "model")
        if self.forced_replace_selector not in replace_selectors:
            raise ValueError(
                "forced replace selector must be one of "
                f"{replace_selectors}")
        if self.forced_operator_best_of < 1:
            raise ValueError("forced operator best-of must be positive")
        if (self.forced_operator_best_of > 1
                and self.forced_operator != "replace"):
            raise ValueError("forced operator best-of requires replace")
        if (self.forced_operator_best_of > 1
                and (self.forced_operator_positive_probe
                     or self.forced_operator_reversible_probe)):
            raise ValueError(
                "forced operator best-of cannot be combined with probes")
        if (self.forced_replace_selector != "base_random"
                and self.forced_operator != "replace"):
            raise ValueError(
                "directed replace selector requires forced replace")
        if (self.forced_replace_selector != "base_random"
                and (self.forced_operator_best_of > 1
                     or self.forced_operator_positive_probe
                     or self.forced_operator_reversible_probe)):
            raise ValueError(
                "directed replace selector cannot be combined with best-of "
                "or probes")
        self.forced_operator_candidate_evaluations = 0
        self.forced_operator_probe_completed = False
        self.forced_operator_probe_positive = False
        self.forced_operator_probe_delta_hv = None
        self.forced_operator_probe_deltas_hv = []
        self.forced_operator_probe_attempts = 0
        self.forced_operator_probe_next_event = 0
        self.forced_operator_probe_fallbacks = 0
        self.forced_operator_probe_evaluations = 0
        if self.forced_operator_min_generation is not None:
            if self.forced_operator is None:
                raise ValueError(
                    "forced operator schedule requires forced M operator")
            if self.forced_operator_min_generation < 0:
                raise ValueError(
                    "forced operator minimum generation cannot be negative")
            if self.model_choice_event is not None:
                raise ValueError(
                    "forced operator schedule and model_choice_event are "
                    "mutually exclusive")
        elif self.forced_operator is not None and self.model_choice_event is None:
            raise ValueError(
                "forced M operator requires model_choice_event or a minimum "
                "generation schedule")
        if self.forced_operator_positive_probe:
            if self.forced_operator_min_generation is None:
                raise ValueError(
                    "positive operator probe requires a generation schedule")
            if self.forced_operator_probability != 1.0:
                raise ValueError(
                    "positive operator probe requires schedule probability 1")
        if (self.forced_operator_reversible_probe
                and not self.forced_operator_positive_probe):
            raise ValueError(
                "reversible operator probe requires positive operator probe")
        if self.forced_operator_probe_max_attempts < 1:
            raise ValueError("operator probe maximum attempts must be positive")
        if self.forced_operator_probe_retry_events < 0:
            raise ValueError("operator probe retry events cannot be negative")
        if self.forced_operator_probe_max_attempts > 1:
            if not self.forced_operator_reversible_probe:
                raise ValueError(
                    "repeated operator probes require a reversible probe")
            if self.forced_operator_probe_retry_events < 1:
                raise ValueError(
                    "repeated operator probes require a positive retry interval")
        if (continuation_event is None) != (continuation_seed is None):
            raise ValueError(
                "M continuation event and seed must be supplied together")
        self.continuation_event = (
            None if continuation_event is None else int(continuation_event))
        self.continuation_seed = (
            None if continuation_seed is None else int(continuation_seed))
        if self.continuation_event is not None and self.continuation_event < 1:
            raise ValueError("M continuation event must be a positive integer")
        self.continuation_reseed_count = 0
        self.mixture_event_count = 0
        self.mutation_control = getattr(
            getattr(trainer, "config", None), "mutation_control", "location")
        self.credit_assignment = getattr(
            getattr(trainer, "config", None), "credit_assignment", "continuous")
        self.candidate_audit_states = int(candidate_audit_states)
        self.candidate_audit_candidates = int(candidate_audit_candidates)
        self.candidate_audit_replicates = int(candidate_audit_replicates)
        if self.candidate_audit_states < 0:
            raise ValueError("candidate_audit_states cannot be negative")
        if self.candidate_audit_candidates < 2:
            raise ValueError("candidate_audit_candidates must be at least 2")
        if self.candidate_audit_replicates < 1:
            raise ValueError("candidate_audit_replicates must be positive")
        if (self.candidate_audit_states
                and self.mutation_control not in ("location", "replace_path")):
            raise ValueError(
                "fixed-state mutation audit requires location or replace_path control")
        self._candidate_audit_pending = None
        self.candidate_audit_state_count = 0
        self.candidate_audit_evaluation_count = 0
        self.candidate_audit_seconds = 0.0
        self.policy_state_count = 0
        self._active_mixture_event_index = None
        self.control_rng = random.Random(private_seed(
            "M-mixture-v2",
            kwargs.get("run_id", args[1] if len(args) > 1 else ""),
            (kwargs.get("seeds") or {}).get("algorithm"),
        ))
        log_names = [
            "mutation_outcomes", "mutation_candidates",
            "mutation_operator_candidates", "mutation_skips",
            "mutation_policy_states", "mutation_operator_intervention_states",
            "generation_summary",
        ]
        if self.candidate_audit_states:
            log_names.extend((
                "mutation_candidate_audit_states",
                "mutation_candidate_audit_outcomes",
            ))
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            preserve_base_position_rng=True, normalizer=archive.normalizer,
            joint_replacement_paths=self.mutation_control == "replace_path",
            log_names=tuple(log_names), **kwargs)
        if (self.model_choice_probability < 1.0
                or self.model_choice_event is not None
                or self.forced_operator_min_generation is not None):
            self.decision_counts["base_mixture_choice"] = 0
        if self.forced_operator is not None:
            self.decision_counts["forced_operator_choice"] = 0
        trainer.observation_provider = self._observation

    def _next_control_branch(self):
        """Return the branch and globally ordered mutation-call index."""
        self.mixture_event_count += 1
        if self.forced_operator_min_generation is not None:
            eligible = self.generation >= self.forced_operator_min_generation
            probability = self.forced_operator_probability
            if self.forced_operator_positive_probe:
                probe_due = bool(
                    self.forced_operator_probe_attempts
                    < self.forced_operator_probe_max_attempts
                    and (
                        self.forced_operator_probe_attempts == 0
                        or self.mixture_event_count
                        >= self.forced_operator_probe_next_event))
                use_model = bool(
                    eligible
                    and (self.forced_operator_probe_positive or probe_due))
            else:
                use_model = bool(
                    eligible and (
                        probability == 1.0
                        or (probability > 0.0
                            and self.control_rng.random() < probability)))
        elif self.model_choice_event is not None:
            use_model = self.mixture_event_count == self.model_choice_event
        else:
            use_model = (
                True if self.model_choice_probability == 1.0 else
                False if self.model_choice_probability == 0.0 else
                self.control_rng.random() < self.model_choice_probability
            )
        return use_model, self.mixture_event_count

    def choose_mutation_candidate(self, candidates, context):
        forced = (
            self.forced_candidate_index
            if self._active_mixture_event_index == self.model_choice_event
            else None)
        return self.trainer.choose(
            candidates, context, forced_action_index=forced)

    def _reseed_continuation_if_due(self, mixture_event_index):
        """Start a matched post-intervention random stream once."""
        if (self.continuation_event is None
                or mixture_event_index != self.continuation_event):
            return False
        if self.continuation_reseed_count:
            raise RuntimeError("M continuation random stream reseeded twice")
        random.seed(self.continuation_seed)
        np.random.seed(self.continuation_seed % (2**32 - 1))
        self.continuation_reseed_count += 1
        return True

    def _directed_replace_candidate(
            self, ind, batches, reliable_options):
        """Choose a joint batch/path replacement without objective evaluation."""
        rows = [
            row for row in replacement_path_candidates(
                ind, batches, reliable_options)
            if row.get("eligible") and row.get("path") is not None
        ]
        if not rows:
            return None, {"eligible_candidate_count": 0}

        def finite(value, default=0.0):
            value = float(value)
            return value if math.isfinite(value) else float(default)

        raw = []
        for row in rows:
            quantity = finite(row.get("batch_quantity", 0.0))
            raw.append([
                quantity * finite(row["replacement_delta_cost_per_teu"]),
                quantity * finite(
                    row["replacement_delta_emission_per_teu"]),
                finite(row["replacement_delta_nominal_time_h"]),
                finite(row["replacement_max_lateness_h"], 1e9),
                -finite(row["replacement_on_time_probability"]),
            ])
        values = np.asarray(raw, dtype=float)
        minima = values.min(axis=0)
        spans = values.max(axis=0) - minima
        spans[spans <= 1e-15] = 1.0
        scaled = (values - minima) / spans

        selector = self.forced_replace_selector
        forced_index = int(self.decision_counts.get(
            "forced_operator_choice", 0))
        criterion = None
        if selector == "joint_extreme_cycle":
            criterion = ("cost", "emission", "time", "reliability")[
                forced_index % 4]
            column = {
                "cost": 0, "emission": 1, "time": 2,
                "reliability": 3,
            }[criterion]
            indices = list(range(len(rows)))
            if criterion == "reliability":
                key = lambda index: (
                    values[index, 3], values[index, 4],
                    values[index, 2],
                    int(rows[index]["target"].batch_id),
                    str(rows[index].get("path_id")))
            else:
                key = lambda index: (
                    values[index, column],
                    int(rows[index]["target"].batch_id),
                    str(rows[index].get("path_id")))
            selected_index = min(indices, key=key)
        elif selector == "joint_balanced":
            scores = (
                scaled[:, 0] + scaled[:, 1]
                + (scaled[:, 2] + scaled[:, 3] + scaled[:, 4]) / 3.0)
            selected_index = min(
                range(len(rows)),
                key=lambda index: (
                    float(scores[index]),
                    int(rows[index]["target"].batch_id),
                    str(rows[index].get("path_id"))))
        elif selector == "joint_pareto_cycle":
            time_risk = (
                scaled[:, 2] + scaled[:, 3] + scaled[:, 4]) / 3.0
            objectives = np.column_stack(
                (scaled[:, 0], scaled[:, 1], time_risk))
            front = []
            for index, vector in enumerate(objectives):
                dominated = np.any(
                    np.all(objectives <= vector, axis=1)
                    & np.any(objectives < vector, axis=1))
                if not dominated:
                    front.append(index)
            front.sort(key=lambda index: (
                int(rows[index]["target"].batch_id),
                str(rows[index].get("path_id"))))
            selected_index = front[forced_index % len(front)]
        else:
            raise RuntimeError(
                f"unsupported directed replace selector: {selector}")

        row = rows[selected_index]
        return row, {
            "eligible_candidate_count": len(rows),
            "criterion": criterion,
            "selected_batch_id": int(row["target"].batch_id),
            "selected_path_id": str(row.get("path_id")),
            "selected_proxy_vector": raw[selected_index],
            "selected_scaled_vector": scaled[selected_index].tolist(),
        }

    def _mutate_forced_operator(
            self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options,
            mixture_event_index, kwargs):
        """Force one operator while leaving Base batch/target logic intact."""
        if self.forced_operator_best_of > 1:
            return self._mutate_forced_operator_best_of(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options,
                mixture_event_index, kwargs)
        self._inside_mutation = True
        before_id = fingerprint(ind)
        archive_hv_before = float(self.archive.hypervolume)
        operator_probe_attempt = bool(
            self.forced_operator_positive_probe
            and not self.forced_operator_probe_positive
            and self.forced_operator_probe_attempts
            < self.forced_operator_probe_max_attempts)
        reversible_probe = bool(
            self.forced_operator_reversible_probe
            and operator_probe_attempt)
        probe_parent = deepcopy(ind) if reversible_probe else None
        candidates = self._operator_candidates(
            ind, batches, path_lib, tt_dict, arc_lookup, reliable_options)
        operator_index = OPERATORS.index(self.forced_operator)
        self.append("mutation_operator_intervention_states", {
            "run_id": self.run_id,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "mixture_event_index": mixture_event_index,
            "parent_id": before_id,
            "forced_operator": self.forced_operator,
            "operator_candidates": candidates,
        })

        # Base draws the batch before the operator. Consume those shadow draws
        # even when the directed selector replaces the joint batch/path choice.
        base_python_state = random.getstate() if reversible_probe else None
        base_numpy_state = np.random.get_state() if reversible_probe else None
        shadow_base_batch = random.choice(batches)
        shadow_base_operator = base.sample_operator()
        batch = shadow_base_batch
        mutation_target = None
        selector_metadata = None
        model_decision = None
        if self.forced_replace_selector == "model":
            replacement_candidates = replacement_path_candidates(
                ind, batches, reliable_options)
            legal_target_count = sum(
                bool(row["eligible"]) for row in replacement_candidates)
            for candidate in replacement_candidates:
                candidate["candidate_operator"] = "replace"
                candidate["operator_legal_target_count"] = legal_target_count
                candidate["operator_legal_target_fraction"] = (
                    legal_target_count / max(1, len(replacement_candidates)))
            context = {
                "operator": "replace",
                "generation": self.generation,
                "phase": self.phase,
                "K": self.instance.get("K"),
                "deadline_window_alpha": self.instance.get(
                    "deadline_window_alpha"),
                "parent_objectives_available": False,
                "feasible_before": None,
                "violation_before": None,
                "violation_breakdown_before": None,
                "candidate_count": len(replacement_candidates),
                "q90_cost_before": None,
                "q90_emission_before": None,
                "q90_makespan_before": None,
            }
            self.current_individual = ind
            try:
                model_decision = self.trainer.choose(
                    replacement_candidates, context)
            finally:
                self.current_individual = None
            self.inference_seconds += float(
                model_decision.metadata.get("inference_seconds", 0.0))
            if model_decision.chosen_index is not None:
                selected = replacement_candidates[
                    model_decision.chosen_index]
                batch_by_id = {item.batch_id: item for item in batches}
                batch = batch_by_id[selected["target"].batch_id]
                mutation_target = selected["target"]
                selector_metadata = {
                    "eligible_candidate_count": legal_target_count,
                    "selected_candidate_index": int(
                        model_decision.chosen_index),
                    "selected_batch_id": int(
                        selected["target"].batch_id),
                    "selected_path_id": str(selected.get("path_id")),
                }
        elif self.forced_replace_selector != "base_random":
            selected, selector_metadata = self._directed_replace_candidate(
                ind, batches, reliable_options)
            if selected is not None:
                batch_by_id = {item.batch_id: item for item in batches}
                batch = batch_by_id[selected["target"].batch_id]
                mutation_target = selected["target"]
        previous_suppress_archive = self._suppress_archive
        evaluations_before_probe = self.evaluations
        try:
            if reversible_probe:
                self._suppress_archive = True
            ok = base.apply_mutation_op(
                ind, self.forced_operator, batch, path_lib, tt_dict, arc_lookup,
                reliable_options=reliable_options, target=mutation_target)
            if ok:
                base.repair_missing_allocations(ind, batches, path_lib)
                base.evaluate_individual(
                    ind, batches, arcs, tt_dict, waiting_cost, waiting_emission,
                    **kwargs)
        finally:
            self._suppress_archive = previous_suppress_archive
            self._inside_mutation = False
        after_id = fingerprint(ind)
        changed = after_id != before_id
        if reversible_probe and ok:
            preview = self.archive.preview_many([(
                ind.objectives, bool(ind.feasible), after_id)])
            immediate_archive_delta = float(preview.delta_hv)
            self.forced_operator_probe_evaluations += (
                self.evaluations - evaluations_before_probe)
        else:
            archive_hv_after = float(self.archive.hypervolume)
            immediate_archive_delta = archive_hv_after - archive_hv_before
        if operator_probe_attempt:
            self.forced_operator_probe_attempts += 1
            self.forced_operator_probe_completed = True
            self.forced_operator_probe_delta_hv = immediate_archive_delta
            self.forced_operator_probe_deltas_hv.append(
                immediate_archive_delta)
            self.forced_operator_probe_positive = (
                immediate_archive_delta > 0.0)
            if not self.forced_operator_probe_positive:
                self.forced_operator_probe_next_event = (
                    mixture_event_index
                    + self.forced_operator_probe_retry_events + 1)
        probe_reverted_to_base = bool(
            reversible_probe and immediate_archive_delta <= 0.0)
        executed_operator = self.forced_operator
        if probe_reverted_to_base:
            ind.__dict__.clear()
            ind.__dict__.update(deepcopy(probe_parent.__dict__))
            random.setstate(base_python_state)
            np.random.set_state(base_numpy_state)
            self._inside_mutation = True
            self.controls_mutation = False
            try:
                executed_operator, ok = base.mutate_fixed(
                    ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                    waiting_cost, waiting_emission,
                    reliable_options=reliable_options, **kwargs)
            finally:
                self.controls_mutation = True
                self._inside_mutation = False
            if executed_operator != shadow_base_operator:
                raise RuntimeError(
                    "reversible M probe failed to replay the Base operator")
            self.forced_operator_probe_fallbacks += 1
            after_id = fingerprint(ind)
            changed = after_id != before_id
        elif reversible_probe:
            self.archive.observe(
                ind.objectives, bool(ind.feasible), after_id)
        self.total_attempts += 1
        self.total_effective += int(bool(ok) and changed)
        self.decision_counts["forced_operator_choice"] += 1
        policy_reward = hv_reward(immediate_archive_delta, self.reward_config)
        policy_action = bool(
            model_decision is not None
            and model_decision.metadata.get("decision_kind") == "model_choice")
        if model_decision is not None and self.trainer.pending is not None:
            self.trainer.observe_outcome(policy_reward, done=True)
        if policy_action:
            self.reward_total += policy_reward
            self.positive_reward_actions += int(policy_reward > 0.0)
        self.append("mutation_outcomes", {
            "run_id": self.run_id,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "decision_kind": "forced_operator_choice",
            "control_branch": "forced_operator",
            "model_choice_probability": self.model_choice_probability,
            "model_choice_event": self.model_choice_event,
            "forced_candidate_index": None,
            "forced_operator": self.forced_operator,
            "forced_replace_selector": self.forced_replace_selector,
            "mixture_event_index": mixture_event_index,
            "selected_candidate_index": (
                model_decision.chosen_index
                if model_decision is not None else operator_index),
            "operator": executed_operator,
            "shadow_base_operator": shadow_base_operator,
            "different_from_shadow_base_operator": (
                self.forced_operator != shadow_base_operator),
            "batch_id": batch.batch_id,
            "shadow_base_batch_id": shadow_base_batch.batch_id,
            "replace_selector_metadata": selector_metadata,
            "mutation_success": bool(ok),
            "decision_changed": bool(changed),
            "immediate_archive_delta": immediate_archive_delta,
            "operator_probe_completed": self.forced_operator_probe_completed,
            "operator_probe_positive": self.forced_operator_probe_positive,
            "probe_reverted_to_base": probe_reverted_to_base,
            "parent_id": before_id,
            "child_id": after_id,
            "policy_metadata": (
                model_decision.metadata if model_decision is not None else {
                    "forced_operator": self.forced_operator,
                    "forced_action_index": operator_index,
                }),
            "reward": float(policy_reward) if policy_action else 0.0,
            "reward_hv_delta": (
                float(immediate_archive_delta) if policy_action else 0.0),
            "reward_recorded_for_policy": policy_action,
        })
        if self.phase == "offspring":
            self._record_pending_offspring(ind)
        self._reseed_continuation_if_due(mixture_event_index)
        return executed_operator, bool(ok)

    def _mutate_forced_operator_best_of(
            self, ind, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options,
            mixture_event_index, kwargs):
        """Evaluate multiple budget-counted replacements and keep the best."""
        if self.forced_operator != "replace":
            raise RuntimeError("best-of mutation requires replace")
        self._inside_mutation = True
        before_id = fingerprint(ind)
        parent = deepcopy(ind)
        candidates = self._operator_candidates(
            ind, batches, path_lib, tt_dict, arc_lookup, reliable_options)
        operator_index = OPERATORS.index(self.forced_operator)
        self.append("mutation_operator_intervention_states", {
            "run_id": self.run_id,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "mixture_event_index": mixture_event_index,
            "parent_id": before_id,
            "forced_operator": self.forced_operator,
            "forced_operator_best_of": self.forced_operator_best_of,
            "operator_candidates": candidates,
        })

        # Preserve Base's batch-before-operator draw order.  The internal
        # replace target/path draws are deliberately repeated for the extra
        # candidates and every successful candidate evaluation consumes the
        # shared real-objective budget.
        batch = random.choice(batches)
        shadow_base_operator = base.sample_operator()
        remaining = (
            self.forced_operator_best_of if self.budget is None else
            max(0, int(self.budget) - self.evaluations - 1))
        maximum_evaluations = min(self.forced_operator_best_of, remaining)
        evaluated = []
        seen_fingerprints = set()
        previous_suppress_archive = self._suppress_archive
        try:
            self._suppress_archive = True
            # A few extra draws make the requested K candidates genuinely
            # distinct when the stochastic replace operator repeats a path.
            maximum_draws = max(self.forced_operator_best_of,
                                5 * self.forced_operator_best_of)
            for attempt in range(maximum_draws):
                if len(evaluated) >= maximum_evaluations:
                    break
                candidate = deepcopy(parent)
                ok = base.apply_mutation_op(
                    candidate, self.forced_operator, batch, path_lib, tt_dict,
                    arc_lookup, reliable_options=reliable_options, target=None)
                if not ok:
                    continue
                base.repair_missing_allocations(candidate, batches, path_lib)
                candidate_id = fingerprint(candidate)
                if candidate_id in seen_fingerprints:
                    continue
                seen_fingerprints.add(candidate_id)
                if not self.has_budget(1):
                    break
                base.evaluate_individual(
                    candidate, batches, arcs, tt_dict, waiting_cost,
                    waiting_emission, **kwargs)
                preview = self.archive.preview_many([(
                    candidate.objectives, bool(candidate.feasible),
                    candidate_id)])
                try:
                    normalized = self.archive.normalizer.transform(
                        candidate.objectives)
                    quality = -sum(normalized)
                except ValueError:
                    quality = float("-inf")
                evaluated.append({
                    "attempt": attempt + 1,
                    "individual": candidate,
                    "fingerprint": candidate_id,
                    "feasible": bool(candidate.feasible),
                    "archive_delta_hv": float(preview.delta_hv),
                    "quality_tiebreak": float(quality),
                    "objectives": [
                        float(value) for value in candidate.objectives],
                })
        finally:
            self._suppress_archive = previous_suppress_archive
            self._inside_mutation = False

        self.forced_operator_candidate_evaluations += len(evaluated)
        if evaluated:
            best = max(evaluated, key=lambda row: (
                row["feasible"], row["archive_delta_hv"],
                row["quality_tiebreak"], -row["attempt"]))
            ind.__dict__.clear()
            ind.__dict__.update(deepcopy(best["individual"].__dict__))
            immediate_archive_delta = float(self.archive.observe(
                ind.objectives, bool(ind.feasible), best["fingerprint"])
            )
            ok = True
        else:
            best = None
            immediate_archive_delta = 0.0
            ok = False

        after_id = fingerprint(ind)
        changed = after_id != before_id
        self.total_attempts += 1
        self.total_effective += int(ok and changed)
        self.decision_counts["forced_operator_choice"] += 1
        self.append("mutation_outcomes", {
            "run_id": self.run_id,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "decision_kind": "forced_operator_best_of_choice",
            "control_branch": "forced_operator_best_of",
            "model_choice_probability": self.model_choice_probability,
            "model_choice_event": self.model_choice_event,
            "forced_candidate_index": None,
            "forced_operator": self.forced_operator,
            "forced_operator_best_of": self.forced_operator_best_of,
            "mixture_event_index": mixture_event_index,
            "selected_candidate_index": operator_index,
            "selected_best_of_attempt": (
                None if best is None else best["attempt"]),
            "candidate_evaluation_count": len(evaluated),
            "candidate_diagnostics": [
                {key: value for key, value in row.items()
                 if key != "individual"}
                for row in evaluated
            ],
            "operator": self.forced_operator,
            "shadow_base_operator": shadow_base_operator,
            "different_from_shadow_base_operator": (
                self.forced_operator != shadow_base_operator),
            "batch_id": batch.batch_id,
            "mutation_success": bool(ok),
            "decision_changed": bool(changed),
            "immediate_archive_delta": immediate_archive_delta,
            "parent_id": before_id,
            "child_id": after_id,
            "reward_recorded_for_policy": False,
        })
        if self.phase == "offspring":
            self._record_pending_offspring(ind)
        self._reseed_continuation_if_due(mixture_event_index)
        return self.forced_operator, bool(ok)

    @staticmethod
    def _mean(rows, name):
        values = []
        for row in rows:
            try:
                value = float(row.get(name, 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return sum(values) / len(values) if values else 0.0

    def _operator_candidates(self, ind, batches, path_lib, tt_dict,
                             arc_lookup, reliable_options):
        """Aggregate legal target sets into five operator-level actions."""
        numeric = (
            "batch_quantity", "path_count", "share", "path_length",
            "alternative_mode_count", "available_reliable_options",
            "unallocated_path_count", "path_cost_per_teu",
            "path_emission_per_teu", "path_nominal_time_h",
            "path_total_distance", "path_bottleneck_capacity",
            "alternative_min_cost", "alternative_min_emission",
            "alternative_min_time", "alternative_mean_cost",
            "alternative_mean_emission", "alternative_mean_time",
            "reliable_best_on_time_probability",
            "reliable_mean_on_time_probability",
            "reliable_min_max_lateness_h", "reliable_mean_max_lateness_h",
        )
        result = []
        for operator, prior in zip(OPERATORS, base._FIXED_OP_PROBS):
            targets = candidates_for(
                ind, batches, operator, path_lib, tt_dict, arc_lookup,
                reliable_options)
            legal = [row for row in targets if row["eligible"]]
            source = legal or targets
            candidate = {
                "target": {}, "eligible": bool(legal),
                "selection_probability": float(prior),
                "candidate_operator": operator,
                "operator_stage": True,
                "operator_legal_target_count": len(legal),
                "operator_legal_target_fraction": (
                    len(legal) / max(1, len(targets))),
                "operator_target_count": len(targets),
            }
            candidate.update({name: self._mean(source, name) for name in numeric})
            result.append(candidate)
        return result

    def _choose_operator(self, ind, batches, path_lib, tt_dict, arc_lookup,
                         reliable_options):
        candidates = self._operator_candidates(
            ind, batches, path_lib, tt_dict, arc_lookup, reliable_options)
        context = {
            "component": "M", "operator": "add",
            "decision_stage": "operator", "generation": self.generation,
            "phase": self.phase, "K": self.instance.get("K"),
            "deadline_window_alpha": self.instance.get("deadline_window_alpha"),
            # The crossed child has not been evaluated yet.  Any feasibility
            # fields copied from a mating parent are stale and must not be used.
            "feasible_before": None, "violation_before": None,
        }
        decision = self.trainer.choose(candidates, context)
        self.inference_seconds += float(
            decision.metadata.get("inference_seconds", 0.0))
        event_id = f"{self.run_id}:M-operator:{self.event_count + 1}"
        for index, candidate in enumerate(candidates):
            self.append("mutation_operator_candidates", {
                "event_id": event_id, "candidate_id": index,
                "generation": self.generation, "evaluation": self.evaluations,
                "operator": candidate["candidate_operator"],
                "eligible": candidate["eligible"],
                "operator_target_count": candidate["operator_target_count"],
                "operator_legal_target_count": candidate[
                    "operator_legal_target_count"],
                "operator_legal_target_fraction": candidate[
                    "operator_legal_target_fraction"],
                "baseline_selection_probability": candidate[
                    "selection_probability"],
                "selection_probability": decision.probabilities[index],
                "chosen": index == decision.chosen_index,
                "decision_kind": decision.metadata.get("decision_kind"),
            })
        return decision, candidates

    def evaluate(self, function, ind, *args, **kwargs):
        previous = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        if not self._suppress_archive:
            self.archive.observe(ind.objectives, bool(ind.feasible), fingerprint(ind))
        if self.generation < 0 and self.evaluations > previous and len(self.current_population) < self.pop_size:
            self.current_population.append(ind)
        elif (self.generation >= 0 and self.phase == "offspring"
              and self.evaluations > previous and not self._inside_mutation):
            self._record_pending_offspring(ind)
        return result

    def _record_pending_offspring(self, ind):
        key = fingerprint(ind)
        if key not in self._pending_fingerprints:
            self._pending_fingerprints.add(key)
            self.pending_offspring.append(ind)

    def _observation(self, candidates, context):
        if self.current_individual is None:
            raise RuntimeError("policy requested an observation outside a mutation decision")
        enriched = dict(context)
        enriched["offspring_fraction"] = (
            len(self.pending_offspring) / max(1, self.pop_size))
        # The decision object is a newly crossed child, so any rank/crowding
        # copied from a mating parent is stale and must not enter the state.
        enriched["parent_rank"] = -1
        enriched["parent_crowding"] = 0.0
        observation = build_observation(
            self.current_individual, self.batches_for_observation, candidates, enriched,
            self.current_population, self.pending_offspring,
            self.archive.summary(), self.archive.normalizer, self.budget,
            self.evaluations, self.generations, self.graph_observation,
            getattr(getattr(self.trainer, "config", None),
                    "candidate_representation", "legacy"))
        self.policy_state_count += 1
        candidate_descriptors = []
        for index, candidate in enumerate(candidates):
            target = candidate.get("target", {})
            if is_dataclass(target):
                target = asdict(target)
            elif isinstance(target, dict):
                target = dict(target)
            else:
                raise TypeError("mutation candidate target must be a dataclass or dict")
            candidate_descriptors.append({
                "candidate_id": index,
                "target": target,
                "path_id": candidate.get("path_id"),
                "candidate_operator": candidate.get("candidate_operator"),
                "eligible": bool(candidate.get("eligible")),
            })
        self.append("mutation_policy_states", {
            "policy_state_id": self.policy_state_count,
            "mixture_event_index": self._active_mixture_event_index,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "operator": enriched.get("operator"),
            "decision_stage": enriched.get("decision_stage", "location"),
            "candidates": candidate_descriptors,
            "observation": observation,
        })
        if self._should_capture_candidate_audit(candidates):
            self._candidate_audit_pending = {
                "parent": deepcopy(self.current_individual),
                "candidates": deepcopy(candidates),
                "context": dict(enriched),
                "observation": observation,
                "archive_hv": float(self.archive.hypervolume),
                "archive_size": len(self.archive.points),
            }
        return observation

    def _should_capture_candidate_audit(self, candidates):
        if (not self.candidate_audit_states
                or self.candidate_audit_state_count
                >= self.candidate_audit_states
                or self._candidate_audit_pending is not None):
            return False
        if sum(bool(candidate.get("eligible")) for candidate in candidates) < 2:
            return False
        threshold = (
            (self.candidate_audit_state_count + 1)
            / (self.candidate_audit_states + 1))
        return self.evaluations / max(1, int(self.budget or 1)) >= threshold

    def _candidate_audit_indices(
            self, prior, supervised_prior, trained, chosen, eligible, state_id):
        legal = [index for index, value in enumerate(eligible) if value]
        ordered = []

        def include(index):
            if index in legal and index not in ordered:
                ordered.append(index)

        include(chosen)
        include(max(legal, key=lambda index: (prior[index], -index)))
        include(max(legal, key=lambda index: (
            supervised_prior[index], -index)))
        include(max(legal, key=lambda index: (trained[index], -index)))
        remaining = [index for index in legal if index not in ordered]
        random.Random(private_seed(
            "M-candidate-audit-subset", self.run_id, state_id)).shuffle(remaining)
        for index in remaining:
            include(index)
            if len(ordered) >= self.candidate_audit_candidates:
                break
        return ordered[:self.candidate_audit_candidates]

    def _run_candidate_audit(
            self, batches, path_lib, tt_dict, arc_lookup, arcs,
            waiting_cost, waiting_emission, reliable_options, evaluation_kwargs):
        snapshot = self._candidate_audit_pending
        self._candidate_audit_pending = None
        pending = self.trainer.pending
        if snapshot is None or pending is None:
            return
        probabilities = [float(value) for value in pending["old_probabilities"]]
        observation = snapshot["observation"]
        prior = [float(value) for value in observation.get(
            "base_action_prior", observation["action_prior"])]
        supervised_prior = [float(value)
                            for value in observation["action_prior"]]
        eligible = [bool(value) for value in observation["eligible"]]
        chosen = int(pending["action"])
        state_id = self.candidate_audit_state_count + 1
        audited = self._candidate_audit_indices(
            prior, supervised_prior, probabilities, chosen, eligible, state_id)
        if len(audited) < 2:
            return

        self.candidate_audit_state_count = state_id
        prior_mass = sum(prior[index] for index in audited)
        trained_mass = sum(probabilities[index] for index in audited)
        candidate_descriptors = []
        for index, candidate in enumerate(snapshot["candidates"]):
            target = candidate.get("target")
            candidate_descriptors.append({
                "candidate_id": index,
                "target": asdict(target),
                "eligible": bool(candidate.get("eligible")),
                "path_id": candidate.get("path_id"),
                "baseline_probability": prior[index],
                "supervised_prior_probability": supervised_prior[index],
                "trained_probability": probabilities[index],
                "chosen": index == chosen,
                "audited": index in audited,
            })
        self.append("mutation_candidate_audit_states", {
            "state_id": state_id,
            "instance_id": self.instance.get("instance_id"),
            "generation": self.generation,
            "evaluation": self.evaluations,
            "evaluation_fraction": (
                self.evaluations / max(1, int(self.budget or 1))),
            "operator": snapshot["context"]["operator"],
            "archive_hv": snapshot["archive_hv"],
            "archive_size": snapshot["archive_size"],
            "chosen_candidate_id": chosen,
            "audited_candidate_ids": audited,
            "audited_prior_probability_mass": prior_mass,
            "audited_trained_probability_mass": trained_mass,
            "candidates": candidate_descriptors,
            "observation": observation,
        })

        algorithm_random_state = random.getstate()
        algorithm_numpy_state = np.random.get_state()
        audit_started = time.perf_counter()
        try:
            for replicate in range(self.candidate_audit_replicates):
                completion_seed = private_seed(
                    "M-candidate-audit-completion", self.run_id,
                    state_id, replicate)
                for candidate_index in audited:
                    random.setstate(random.Random(completion_seed).getstate())
                    np.random.seed(completion_seed % (2**32 - 1))
                    candidate = snapshot["candidates"][candidate_index]
                    target = candidate["target"]
                    individual = deepcopy(snapshot["parent"])
                    before = deepcopy(individual)
                    batch = next(
                        item for item in batches
                        if item.batch_id == target.batch_id)
                    mutation_success = base.apply_mutation_op(
                        individual, snapshot["context"]["operator"], batch,
                        path_lib, tt_dict, arc_lookup,
                        reliable_options=reliable_options, target=target)
                    repair = repair_after_mutation(
                        individual, before, batches, path_lib,
                        tt_dict, arc_lookup)
                    # Bypass the active controller: audit evaluations neither
                    # consume the declared search budget nor mutate its archive.
                    base._evaluate_individual_impl(
                        individual, batches, arcs, tt_dict,
                        waiting_cost, waiting_emission, **evaluation_kwargs)
                    self.candidate_audit_evaluation_count += 1
                    individual_fingerprint = fingerprint(individual)
                    diagnostics = self.archive.diagnose_candidate(
                        individual.objectives, bool(individual.feasible),
                        individual_fingerprint)
                    update = self.archive.preview_many([(
                        individual.objectives, bool(individual.feasible),
                        individual_fingerprint)])
                    normalized = None
                    single_point_hv = 0.0
                    try:
                        normalized = list(self.archive.normalizer.transform(
                            individual.objectives))
                        if individual.feasible:
                            single_point_hv = exact_hv_3d(
                                [normalized], self.archive.normalizer.reference)
                    except ValueError:
                        pass
                    self.append("mutation_candidate_audit_outcomes", {
                        "state_id": state_id,
                        "candidate_id": candidate_index,
                        "replicate": replicate,
                        "completion_seed": completion_seed,
                        "operator": snapshot["context"]["operator"],
                        "target": asdict(target),
                        "candidate_features": observation[
                            "candidates"][candidate_index],
                        "global_features": observation["global"],
                        "parent_tokens": observation["parent"],
                        "candidate_graph_tokens": (
                            observation.get("candidate_graph_tokens", [])[candidate_index]
                            if observation.get("candidate_graph_tokens") else None),
                        "baseline_probability": prior[candidate_index],
                        "supervised_prior_probability": supervised_prior[
                            candidate_index],
                        "trained_probability": probabilities[candidate_index],
                        "chosen_by_policy": candidate_index == chosen,
                        "audited_prior_probability_mass": prior_mass,
                        "audited_trained_probability_mass": trained_mass,
                        "mutation_success": bool(mutation_success),
                        "repair": repair,
                        "fingerprint": individual_fingerprint,
                        "feasible": bool(individual.feasible),
                        "normalized_violation": float(
                            individual.normalized_violation),
                        "objectives": [float(value)
                                       for value in individual.objectives],
                        "normalized_objectives": normalized,
                        "single_point_hv": float(single_point_hv),
                        "archive_delta_hv": float(update.delta_hv),
                        "archive_diagnostics": diagnostics,
                    })
        finally:
            random.setstate(algorithm_random_state)
            np.random.set_state(algorithm_numpy_state)
            self.candidate_audit_seconds += time.perf_counter() - audit_started

    def mutate(self, ind, batches, path_lib, tt_dict, arc_lookup,
               arcs, waiting_cost, waiting_emission, reliable_options=None, **kwargs):
        # Preserve the original crossover -> mutation -> repair -> evaluation
        # order.  In particular, do not evaluate the crossed child here.
        self._candidate_audit_pending = None
        use_model, mixture_event_index = self._next_control_branch()
        self._active_mixture_event_index = mixture_event_index
        if not use_model:
            self.base_control_events += 1
            before = fingerprint(ind)
            self._inside_mutation = True
            # Re-enter Base's exact implementation without recursively routing
            # it back to this controller.  Its random batch, operator, target,
            # repair, and evaluation stream are therefore unchanged.
            self.controls_mutation = False
            try:
                result = base.mutate_fixed(
                    ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                    waiting_cost, waiting_emission,
                    reliable_options=reliable_options, **kwargs)
            finally:
                self.controls_mutation = True
                self._inside_mutation = False
            changed = fingerprint(ind) != before
            after = fingerprint(ind)
            self.total_attempts += 1
            self.total_effective += int(changed)
            self.decision_counts["base_mixture_choice"] += 1
            self.append("mutation_outcomes", {
                "run_id": self.run_id,
                "instance_id": self.instance.get("instance_id"),
                "generation": self.generation,
                "evaluation": self.evaluations,
                "decision_kind": "base_mixture_choice",
                "control_branch": "paired_base",
                "model_choice_probability": self.model_choice_probability,
                "model_choice_event": self.model_choice_event,
                "forced_candidate_index": self.forced_candidate_index,
                "mixture_event_index": mixture_event_index,
                "operator": result[0],
                "decision_changed": bool(changed),
                "parent_id": before,
                "child_id": after,
                "reward_recorded_for_policy": False,
            })
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            self._reseed_continuation_if_due(mixture_event_index)
            return result

        self.model_control_events += 1
        if self.forced_operator is not None:
            return self._mutate_forced_operator(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options,
                mixture_event_index, kwargs)
        self._inside_mutation = True
        pending_before = len(self.pending)
        self.current_individual = ind
        self._suppress_archive = True
        started = time.perf_counter()
        hierarchical = self.mutation_control == "operator_location"
        selected_operator = None
        operator_decision_kind = None
        try:
            if hierarchical:
                self.trainer.begin_sequence()
                operator_decision, operator_candidates = self._choose_operator(
                    ind, batches, path_lib, tt_dict, arc_lookup,
                    reliable_options)
                operator_decision_kind = operator_decision.metadata.get(
                    "decision_kind")
                if operator_decision.chosen_index is None:
                    # Match the two Base stream draws that the delegated
                    # location mutation would otherwise consume.
                    random.choice(batches)
                    base.sample_operator()
                    self.trainer.end_sequence(0.0, done=True)
                    self.no_candidate += 1
                    self._reseed_continuation_if_due(mixture_event_index)
                    return "none", False
                selected_operator = operator_candidates[
                    operator_decision.chosen_index]["candidate_operator"]
                if self.trainer.pending is not None:
                    self.trainer.observe_outcome(0.0, done=False)
            result = super().mutate(
                ind, batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options=reliable_options,
                operator_override=selected_operator, **kwargs)
        except Exception:
            if hierarchical and self.trainer.sequence_depth:
                self.trainer.sequence_depth = 0
                self.trainer.sequence_policy_actions = 0
                self.trainer.pending = None
            raise
        finally:
            self._suppress_archive = False
            self.current_individual = None
            self._inside_mutation = False
        if (self._candidate_audit_pending is not None
                and self.trainer.pending is not None):
            # The observation provider captures the pre-policy state.  The
            # supervised prior is applied by PPOTrainer afterwards, so replace
            # the audit copy with the exact observation used by the actor.
            self._candidate_audit_pending["observation"] = deepcopy(
                self.trainer.pending["observation"])
        if self._candidate_audit_pending is not None:
            self._run_candidate_audit(
                batches, path_lib, tt_dict, arc_lookup, arcs,
                waiting_cost, waiting_emission, reliable_options, kwargs)
        if len(self.pending) == pending_before:
            # no eligible candidate: no policy action and no PPO update
            if hierarchical and self.trainer.sequence_depth:
                self.trainer.end_sequence(0.0, done=True)
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            self._reseed_continuation_if_due(mixture_event_index)
            return result
        row = self.pending[-1][1]
        row.update(
            model_choice_event=self.model_choice_event,
            mixture_event_index=mixture_event_index)
        if row.get("base_passthrough"):
            # The probability gate fired, but the paired Base operator was not
            # replace.  MutationLogger has executed the exact Base continuation;
            # it is not a policy action and must not alter the reward archive.
            self.base_passthrough_events += 1
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            self._reseed_continuation_if_due(mixture_event_index)
            return result
        individual_fingerprint = fingerprint(ind)
        reward_diagnostics = self.archive.diagnose_candidate(
            ind.objectives, bool(ind.feasible), individual_fingerprint)
        update = self.archive.observe_many([
            (ind.objectives, bool(ind.feasible), individual_fingerprint),
        ])
        delta_hv = update.delta_hv
        reward = hv_reward(delta_hv, self.reward_config)
        policy_actions = (
            self.trainer.sequence_policy_actions if hierarchical else
            int(row.get("decision_kind") == "model_choice"))
        row.update(reward=float(reward), reward_hv_delta=float(delta_hv),
                   reward_diagnostics=reward_diagnostics,
                   reward_positive_hv=bool(delta_hv > 0.0),
                   reward_archive_changed_without_hv=bool(
                       reward_diagnostics["would_change_archive"]
                       and delta_hv <= 0.0),
                   reward_archive_hv_before=update.hv_before,
                   reward_archive_hv_after=update.hv_after,
                   reward_archive_size=update.archive_size,
                   reward_recorded_for_policy=bool(policy_actions),
                   mutation_control=self.mutation_control,
                   operator_decision_kind=operator_decision_kind,
                   policy_actions_in_operation=policy_actions,
                   action_wall_seconds=time.perf_counter() - started)
        if policy_actions:
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0)
        if hierarchical:
            self.trainer.end_sequence(reward, done=True)
        else:
            self.trainer.observe_outcome(
                reward, done=self.credit_assignment == "operation_terminal")
        if self.phase == "offspring":
            self._record_pending_offspring(ind)
        self._reseed_continuation_if_due(mixture_event_index)
        return result

    def flush_events(self, population):
        # This hook is called immediately after P+Q environmental selection and
        # after boost replacement.  It is the exact point at which pending Q is
        # folded into the next population state.
        super().flush_events(population)
        self.current_population = list(population)
        self.pending_offspring.clear()
        self._pending_fingerprints.clear()

    def log_generation(self, population, boost):
        self.current_population = list(population)
        super().log_generation(population, boost)
        self.log_search_progress(population, "environmental_selection")

    def finish_episode(self):
        self.trainer.finish()
        self.log_policy_updates()

    def summary(self):
        return {"reward_total": self.reward_total,
                "positive_reward_actions": self.positive_reward_actions,
                "reward_archive_hv": self.archive.hypervolume,
                "reward_archive_size": len(self.archive.points),
                "decision_counts": dict(self.decision_counts),
                "ppo_updates": self.trainer.updates - self._ppo_updates_at_start,
                "ppo_updates_before_run": self._ppo_updates_at_start,
                "ppo_updates_total": self.trainer.updates,
                "mutation_control": self.mutation_control,
                "credit_assignment": self.credit_assignment,
                "model_choice_probability": self.model_choice_probability,
                "model_choice_event": self.model_choice_event,
                "forced_candidate_index": self.forced_candidate_index,
                "forced_operator": self.forced_operator,
                "forced_operator_min_generation": (
                    self.forced_operator_min_generation),
                "forced_operator_probability": self.forced_operator_probability,
                "forced_operator_positive_probe": (
                    self.forced_operator_positive_probe),
                "forced_operator_reversible_probe": (
                    self.forced_operator_reversible_probe),
                "forced_operator_probe_completed": (
                    self.forced_operator_probe_completed),
                "forced_operator_probe_positive": (
                    self.forced_operator_probe_positive),
                "forced_operator_probe_delta_hv": (
                    self.forced_operator_probe_delta_hv),
                "forced_operator_probe_deltas_hv": list(
                    self.forced_operator_probe_deltas_hv),
                "forced_operator_probe_attempts": (
                    self.forced_operator_probe_attempts),
                "forced_operator_probe_max_attempts": (
                    self.forced_operator_probe_max_attempts),
                "forced_operator_probe_retry_events": (
                    self.forced_operator_probe_retry_events),
                "forced_operator_probe_next_event": (
                    self.forced_operator_probe_next_event),
                "forced_operator_probe_fallbacks": (
                    self.forced_operator_probe_fallbacks),
                "forced_operator_probe_evaluations": (
                    self.forced_operator_probe_evaluations),
                "forced_operator_best_of": self.forced_operator_best_of,
                "forced_replace_selector": self.forced_replace_selector,
                "forced_operator_candidate_evaluations": (
                    self.forced_operator_candidate_evaluations),
                "continuation_event": self.continuation_event,
                "continuation_seed": self.continuation_seed,
                "continuation_reseed_count": self.continuation_reseed_count,
                "mixture_event_count": self.mixture_event_count,
                "model_control_events": self.model_control_events,
                "base_control_events": self.base_control_events,
                "base_passthrough_events": self.base_passthrough_events,
                "candidate_audit_state_count": self.candidate_audit_state_count,
                "candidate_audit_evaluation_count": (
                    self.candidate_audit_evaluation_count),
                "candidate_audit_seconds": self.candidate_audit_seconds,
                "feature_construction_seconds": self.feature_construction_seconds,
                "ppo_update_seconds": self.ppo_update_seconds}
