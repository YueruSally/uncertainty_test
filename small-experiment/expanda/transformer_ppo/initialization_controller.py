"""PPO controller for sequential distinct-path initialization."""
from __future__ import annotations

import random
import time

import baseline_uncertainty as base
from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_policy import Decision

from .archive import RewardArchive, RewardConfig, hv_reward
from .initialization_observations import build_initialization_observation
from .initialization_rule_features import (
    path_rule_score,
    validate_rule_normalization,
)
from .v2_controls import private_seed


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


class InitializationPPOController(MutationLogger):
    """Change only Base's random path identities during initialization."""

    controls_mutation = False
    controls_crossover = False
    controls_initialization = True

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size,
                 graph_observation=None, rule_normalization=None,
                 rule_top_k=None, model_choice_event=None,
                 forced_candidate_index=None, **kwargs):
        self.trainer = trainer
        self._ppo_updates_at_start = int(trainer.updates)
        self.archive = archive
        self.reward_config = reward_config
        self.batches_for_observation = batches
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.graph_observation = graph_observation
        if (rule_normalization is None) != (rule_top_k is None):
            raise ValueError(
                "I rule normalization and top-k must be supplied together")
        if rule_top_k is not None and int(rule_top_k) < 2:
            raise ValueError("I rule top-k must be at least 2")
        self.rule_normalization = (
            None if rule_normalization is None else dict(rule_normalization))
        self.rule_top_k = (
            None if rule_top_k is None else int(rule_top_k))
        if self.rule_normalization is not None:
            validate_rule_normalization(self.rule_normalization)
        self.model_choice_event = (
            None if model_choice_event is None else int(model_choice_event))
        self.forced_candidate_index = (
            None if forced_candidate_index is None
            else int(forced_candidate_index))
        if self.model_choice_event is not None:
            if self.model_choice_event < 1:
                raise ValueError("I model choice event must be positive")
            if trainer.training:
                raise ValueError(
                    "I one-shot model choice is a frozen-policy diagnostic")
            if self.rule_top_k is not None:
                raise ValueError(
                    "I one-shot model choice requires the full Base path set")
        if self.forced_candidate_index is not None:
            if self.forced_candidate_index < 0:
                raise ValueError("forced I candidate index cannot be negative")
            if self.model_choice_event is None:
                raise ValueError(
                    "forced I candidate index requires model_choice_event")
        self.initialization_decision_count = 0
        self.model_control_events = 0
        self.base_passthrough_events = 0
        self.current_population = []
        self._current_partial = None
        self._current_context = None
        self._active_event = None
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            log_names=("initialization_outcomes", "initialization_candidates",
                       "initialization_count_candidates",
                       "initialization_policy_states",
                       "initialization_skips", "generation_summary"),
            normalizer=archive.normalizer, **kwargs)
        trainer.observation_provider = self._observation

    def evaluate(self, function, ind, *args, **kwargs):
        previous = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        is_active_initialization = (
            self._active_event is not None and id(ind) == self._active_event["individual_id"])
        if not is_active_initialization:
            self.archive.observe(ind.objectives, bool(ind.feasible), fingerprint(ind))
        if self.generation < 0 and self.evaluations > previous:
            self.current_population.append(ind)
        return result

    def _observation(self, candidates, context):
        if self._current_partial is None or self._current_context is None:
            raise RuntimeError("initialization observation requested outside path selection")
        return build_initialization_observation(
            self._current_partial, self.batches_for_observation, candidates,
            self._current_context, self.current_population,
            self.archive.summary(), self.archive.normalizer, self.budget,
            self.evaluations, self.pop_size, self.graph_observation,
            getattr(getattr(self.trainer, "config", None),
                    "candidate_representation", "legacy"))

    def initial_individual(self, batches, path_lib, max_paths=3):
        if self._active_event is not None:
            raise RuntimeError("previous initialized individual was not evaluated")
        started = time.perf_counter()
        individual = base.Individual()
        event_id = f"{self.run_id}:I:{self.event_count + 1}"
        self.event_count += 1
        self.total_attempts += 1
        model_decisions = 0
        selected_rows = []
        differs_from_shadow = False
        sequence_started = False

        try:
            for batch in batches:
                key = (batch.origin, batch.destination, batch.batch_id)
                paths = list(path_lib.get((batch.origin, batch.destination), []))
                if not paths:
                    self.no_candidate += 1
                    self.decision_counts["no_candidate"] += 1
                    self.append("initialization_skips", {
                        "event_id": event_id, "batch_id": batch.batch_id,
                        "reason": "no_legal_path", "decision_kind": "no_candidate",
                    })
                    continue

                maximum_path_count = min(int(max_paths), len(paths))
                shadow_k = random.randint(1, maximum_path_count)
                # Consume Base's exact path-sampling stream even though the
                # policy replaces the result.  All other stochastic choices,
                # including k and split shares, therefore remain paired.
                shadow_paths = random.sample(paths, shadow_k)
                shadow_raw_shares = [random.random() for _ in range(shadow_k)]
                representation = getattr(
                    getattr(self.trainer, "config", None),
                    "candidate_representation", "legacy")
                initialization_control = getattr(
                    getattr(self.trainer, "config", None),
                    "initialization_control", "auto")
                controls_path_count = (
                    initialization_control == "path_count"
                    or (initialization_control == "auto"
                        and representation == "path_v3"))
                if (controls_path_count and maximum_path_count > 1
                        and self.model_choice_event is None):
                    count_candidates = [
                        {"eligible": True, "path_count": count}
                        for count in range(1, maximum_path_count + 1)]
                    count_context = {
                        "component": "I", "batch": batch,
                        "decision_stage": "path_count",
                        "selection_step": 0, "required_path_count": 0,
                        "maximum_path_count": maximum_path_count,
                        "K": self.instance.get("K"),
                        "deadline_window_alpha": self.instance.get(
                            "deadline_window_alpha"),
                    }
                    self._current_partial = individual
                    self._current_context = count_context
                    if not sequence_started:
                        self.trainer.begin_sequence()
                        sequence_started = True
                    elif self.trainer.pending is not None:
                        self.trainer.observe_outcome(0.0, done=False)
                    count_decision = self.trainer.choose(
                        count_candidates, count_context)
                    model_decisions += 1
                    k = count_candidates[
                        count_decision.chosen_index]["path_count"]
                    count_kind = count_decision.metadata["decision_kind"]
                    self.decision_counts[count_kind] = (
                        self.decision_counts.get(count_kind, 0) + 1)
                    self.inference_seconds += float(
                        count_decision.metadata.get("inference_seconds", 0.0))
                    for candidate_index, candidate in enumerate(count_candidates):
                        self.append("initialization_count_candidates", {
                            "event_id": event_id, "batch_id": batch.batch_id,
                            "candidate_id": candidate_index,
                            "path_count": candidate["path_count"],
                            "maximum_path_count": maximum_path_count,
                            "shadow_base_path_count": shadow_k,
                            "baseline_selection_probability": (
                                1.0 / maximum_path_count),
                            "selection_probability": (
                                count_decision.probabilities[candidate_index]),
                            "chosen": candidate_index == count_decision.chosen_index,
                            "decision_kind": count_kind,
                        })
                else:
                    k = shadow_k
                raw_shares = list(shadow_raw_shares[:k])
                if k > shadow_k:
                    extra_rng = random.Random(private_seed(
                        "I-extra-shares-v3", event_id, batch.batch_id))
                    raw_shares.extend(
                        extra_rng.random() for _ in range(k - shadow_k))
                share_total = sum(raw_shares)
                normalized_shares = [
                    (value / share_total if share_total > 0.0 else 1.0 / k)
                    for value in raw_shares]
                remaining = list(paths)
                selected = []
                shadow_plan = list(shadow_paths)
                for step in range(k):
                    candidate_paths = remaining
                    if self.rule_top_k is not None:
                        candidate_paths = sorted(
                            remaining,
                            key=lambda path: (
                                path_rule_score(path, self.rule_normalization),
                                int(path.path_id),
                                _path_signature(path),
                            ),
                        )[:self.rule_top_k]
                    candidates = [{
                        "eligible": True, "path": path,
                        "path_id": path.path_id,
                        "nodes": list(path.nodes), "modes": list(path.modes),
                        "rule_score": (
                            path_rule_score(path, self.rule_normalization)
                            if self.rule_normalization is not None else None),
                    } for path in candidate_paths]
                    context = {
                        "component": "I", "batch": batch,
                        "selection_step": step,
                        "required_path_count": k,
                        "maximum_path_count": maximum_path_count,
                        "planned_share": normalized_shares[step],
                        "remaining_planned_share": sum(
                            normalized_shares[step:]),
                        "K": self.instance.get("K"),
                        "deadline_window_alpha": self.instance.get(
                            "deadline_window_alpha"),
                    }
                    self._current_partial = individual
                    self._current_context = context
                    decision_event_index = None
                    target_model_choice = False
                    if len(candidates) > 1:
                        self.initialization_decision_count += 1
                        decision_event_index = self.initialization_decision_count
                        target_model_choice = (
                            self.model_choice_event is not None
                            and decision_event_index == self.model_choice_event)
                    shadow_path = shadow_plan[step]
                    shadow_candidate_index = None
                    if self.model_choice_event is not None:
                        shadow_candidate_index = next(
                            index for index, path in enumerate(candidate_paths)
                            if (_path_signature(path)
                                == _path_signature(shadow_path)))
                    if len(candidates) == 1:
                        decision = Decision(
                            0, [1.0], [None], {
                                "no_eligible_candidate": False,
                                "fallback": None,
                                "decision_kind": "single_candidate",
                            })
                    elif self.model_choice_event is not None and not target_model_choice:
                        probabilities = [0.0] * len(candidates)
                        probabilities[shadow_candidate_index] = 1.0
                        decision = Decision(
                            shadow_candidate_index, probabilities,
                            [None] * len(candidates), {
                                "no_eligible_candidate": False,
                                "fallback": "exact_base_passthrough",
                                "decision_kind": "base_passthrough",
                                "forced_action_index": None,
                            })
                        self.base_passthrough_events += 1
                    else:
                        if not sequence_started:
                            self.trainer.begin_sequence()
                            sequence_started = True
                        elif self.trainer.pending is not None:
                            self.trainer.observe_outcome(0.0, done=False)
                        if target_model_choice:
                            decision = self.trainer.choose(
                                candidates, context,
                                forced_action_index=(
                                    self.forced_candidate_index))
                        else:
                            decision = self.trainer.choose(candidates, context)
                        model_decisions += 1
                        if target_model_choice:
                            self.model_control_events += 1
                            pending = self.trainer.pending
                            observation = (
                                pending.get("observation")
                                if isinstance(pending, dict) else
                                getattr(self.trainer, "last_observation", None))
                            if observation is None:
                                raise RuntimeError(
                                    "I target policy observation was not retained")
                            self.append("initialization_policy_states", {
                                "event_id": event_id,
                                "decision_event_index": decision_event_index,
                                "batch_id": batch.batch_id,
                                "selection_step": step,
                                "required_path_count": k,
                                "candidate_path_ids": [
                                    str(path.path_id) for path in candidate_paths],
                                "shadow_base_candidate_index": (
                                    shadow_candidate_index),
                                "forced_candidate_index": (
                                    self.forced_candidate_index),
                                "observation": observation,
                            })
                    kind = decision.metadata["decision_kind"]
                    self.decision_counts[kind] = self.decision_counts.get(kind, 0) + 1
                    self.inference_seconds += float(
                        decision.metadata.get("inference_seconds", 0.0))
                    chosen_index = decision.chosen_index
                    if chosen_index is None:
                        raise RuntimeError("non-empty initialization candidates were rejected")
                    chosen = candidate_paths[chosen_index]
                    if target_model_choice and chosen_index != shadow_candidate_index:
                        chosen_signature = _path_signature(chosen)
                        for future_step in range(step + 1, len(shadow_plan)):
                            if (_path_signature(shadow_plan[future_step])
                                    == chosen_signature):
                                shadow_plan[future_step] = shadow_path
                                break
                    remaining.remove(chosen)
                    selected.append(chosen)
                    individual.od_allocations[key] = [
                        base.PathAllocation(
                            path=path,
                            share=(normalized_shares[index]
                                   if representation == "path_v3" else
                                   raw_shares[index]))
                        for index, path in enumerate(selected)
                    ]
                    for candidate_index, candidate in enumerate(candidates):
                        self.append("initialization_candidates", {
                            "event_id": event_id, "batch_id": batch.batch_id,
                            "decision_stage": "path",
                            "decision_event_index": decision_event_index,
                            "selection_step": step, "required_path_count": k,
                            "planned_share": normalized_shares[step],
                            "candidate_id": candidate_index,
                            "path_id": candidate["path_id"],
                            "nodes": candidate["nodes"], "modes": candidate["modes"],
                            "rule_score": candidate["rule_score"],
                            "rule_prefilter_top_k": self.rule_top_k,
                            "candidate_count_before_prefilter": len(remaining) + 1,
                            "decision_kind": kind,
                            "shadow_base_chosen": (
                                candidate_index == shadow_candidate_index),
                            "target_model_choice": target_model_choice,
                            "forced_candidate_index": (
                                self.forced_candidate_index
                                if target_model_choice else None),
                            "baseline_selection_probability": (
                                1.0 / len(candidates)),
                            "selection_probability": decision.probabilities[candidate_index],
                            "chosen": candidate_index == chosen_index,
                        })
                    selected_rows.append({
                        "batch_id": batch.batch_id, "selection_step": step,
                        "decision_event_index": decision_event_index,
                        "required_path_count": k, "chosen_path_id": chosen.path_id,
                        "selected_candidate_index": chosen_index,
                        "shadow_base_candidate_index": shadow_candidate_index,
                        "target_model_choice": target_model_choice,
                        "forced_candidate_index": (
                            self.forced_candidate_index
                            if target_model_choice else None),
                        "planned_share": normalized_shares[step],
                        "decision_kind": kind,
                    })
                individual.od_allocations[key] = base.merge_and_normalize(
                    individual.od_allocations[key])
                # Order matters because Base attaches the independently drawn
                # split shares to sampled paths in order.
                differs_from_shadow = differs_from_shadow or (
                    k != shadow_k or
                    tuple(_path_signature(path) for path in selected)
                    != tuple(_path_signature(path) for path in shadow_paths))
        except Exception:
            if sequence_started and self.trainer.sequence_depth:
                self.trainer.sequence_depth = 0
                self.trainer.sequence_policy_actions = 0
                self.trainer.pending = None
            raise
        finally:
            self._current_partial = None
            self._current_context = None

        self.total_effective += int(differs_from_shadow)
        self._active_event = {
            "event_id": event_id, "individual_id": id(individual),
            "model_decisions": model_decisions,
            "selected_paths": selected_rows,
            "different_from_shadow_base": bool(differs_from_shadow),
            "started": started,
        }
        return individual

    def settle_initialization(self, individual):
        event = self._active_event
        if event is None:
            return
        if id(individual) != event["individual_id"]:
            raise RuntimeError("evaluated individual does not match initialization action")
        reward_diagnostics = self.archive.diagnose_candidate(
            individual.objectives, bool(individual.feasible),
            fingerprint(individual))
        update = self.archive.observe_many([
            (individual.objectives, bool(individual.feasible), fingerprint(individual)),
        ])
        reward = hv_reward(update.delta_hv, self.reward_config)
        if event["model_decisions"]:
            # Each completed individual is one terminal construction sequence;
            # its reward must not leak into decisions for the next individual.
            self.trainer.end_sequence(reward, done=True)
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0.0)
        self.append("initialization_outcomes", {
            **{key: value for key, value in event.items()
               if key not in ("individual_id", "started")},
            "individual_fingerprint": fingerprint(individual),
            "feasible": bool(individual.feasible),
            "objectives": list(individual.objectives),
            "reward": float(reward), "reward_hv_delta": float(update.delta_hv),
            "reward_diagnostics": reward_diagnostics,
            "reward_positive_hv": bool(update.delta_hv > 0.0),
            "reward_archive_changed_without_hv": bool(
                reward_diagnostics["would_change_archive"]
                and update.delta_hv <= 0.0),
            "reward_archive_hv_before": update.hv_before,
            "reward_archive_hv_after": update.hv_after,
            "reward_archive_size": update.archive_size,
            "reward_recorded_for_policy": bool(event["model_decisions"]),
            "evaluation": self.evaluations,
            "action_wall_seconds": time.perf_counter() - event["started"],
        })
        self._active_event = None

    def flush_events(self, population):
        self.current_population = list(population)

    def log_generation(self, population, boost):
        self.generations_completed += 1
        now = time.perf_counter()
        self.append("generation_summary", {
            "run_id": self.run_id, "generation": self.generation,
            "real_objective_evaluation_count": self.evaluations,
            "initialization_attempts_cumulative": self.total_attempts,
            "initializations_different_from_shadow_base": self.total_effective,
            "feasible_ratio": sum(item.feasible for item in population) / len(population),
            "boost_triggered": bool(boost), "runtime_seconds": now - self.start,
            "generation_seconds": now - self._last_generation_time,
            "inference_seconds_cumulative": self.inference_seconds,
            "decision_counts_cumulative": dict(self.decision_counts),
        })
        self._last_generation_time = now
        self.log_search_progress(population, "environmental_selection")

    def finish_episode(self):
        if self._active_event is not None:
            raise RuntimeError("episode ended before initialization reward settlement")
        self.trainer.finish()
        self.log_policy_updates()

    def summary(self):
        return {
            "reward_total": self.reward_total,
            "positive_reward_actions": self.positive_reward_actions,
            "reward_archive_hv": self.archive.hypervolume,
            "reward_archive_size": len(self.archive.points),
            "decision_counts": dict(self.decision_counts),
            "ppo_updates": self.trainer.updates - self._ppo_updates_at_start,
            "ppo_updates_before_run": self._ppo_updates_at_start,
            "ppo_updates_total": self.trainer.updates,
            "rule_prefilter_top_k": self.rule_top_k,
            "rule_prefilter_enabled": self.rule_top_k is not None,
            "initialization_decision_count": self.initialization_decision_count,
            "model_choice_event": self.model_choice_event,
            "model_control_events": self.model_control_events,
            "base_passthrough_events": self.base_passthrough_events,
            "forced_candidate_index": self.forced_candidate_index,
            "feature_construction_seconds": self.feature_construction_seconds,
            "ppo_update_seconds": self.ppo_update_seconds,
        }
