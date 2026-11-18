"""PPO controller for common-node crossover-cut selection."""
from __future__ import annotations

from copy import deepcopy
import math
import random
import time

from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_policy import Decision

from .archive import RewardArchive, RewardConfig, hv_reward, signed_hv_reward
from .crossover import apply_cut, finalize_children, prepare_common_node_crossover
from .crossover_observations import (
    build_crossover_observation,
    build_crossover_shared_observation,
)
from .v2_controls import private_seed, validate_model_choice_probability


class CrossoverPPOController(MutationLogger):
    """Keep Base crossover structure and settle two final children jointly."""

    controls_mutation = False
    controls_crossover = True

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size,
                 graph_observation=None, model_choice_probability=1.0,
                 model_choice_event=None,
                 forced_joint_candidate_index=None,
                 crossover_reward_mode="archive_hv",
                 **kwargs):
        self.trainer = trainer
        self._ppo_updates_at_start = int(trainer.updates)
        self.archive = archive
        self.reward_config = reward_config
        self.batches_for_observation = batches
        self.generations = int(generations)
        self.pop_size = int(pop_size)
        self.graph_observation = graph_observation
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
        self.forced_joint_candidate_index = (
            None if forced_joint_candidate_index is None
            else int(forced_joint_candidate_index))
        if self.forced_joint_candidate_index is not None:
            if self.forced_joint_candidate_index < 0:
                raise ValueError("forced joint candidate index cannot be negative")
            if self.model_choice_event is None:
                raise ValueError(
                    "forced joint candidate index requires model_choice_event")
        if crossover_reward_mode not in (
                "archive_hv", "paired_base_relative",
                "paired_archive_then_quality"):
            raise ValueError(
                "crossover_reward_mode must be archive_hv, paired_base_relative, "
                "or paired_archive_then_quality")
        self.crossover_reward_mode = str(crossover_reward_mode)
        self.counterfactual_evaluations = 0
        self.counterfactual_evaluation_seconds = 0.0
        self.credit_assignment = getattr(
            getattr(trainer, "config", None), "credit_assignment", "continuous")
        self.control_rng = random.Random(private_seed(
            "C-mixture-v2", kwargs.get("run_id", args[1] if len(args) > 1 else ""),
            (kwargs.get("seeds") or {}).get("algorithm")))
        self.current_population = []
        self.pending_offspring = []
        self._pending_fingerprints = set()
        self._current_parents = None
        self._current_candidates = None
        self._shared_observation = None
        self._last_observation = None
        self.shared_observation_builds = 0
        self._active_event = None
        self._pending_child_ids = set()
        self._settled_outcomes = []
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        self.model_control_events = 0
        self.base_control_events = 0
        self.joint_action_reward_stats = {
            "base_passthrough": {"operations": 0, "positive": 0,
                                 "reward_total": 0.0, "raw_hv_total": 0.0},
            "single_cut_override": {"operations": 0, "positive": 0,
                                    "reward_total": 0.0, "raw_hv_total": 0.0},
        }
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            log_names=("crossover_outcomes", "crossover_candidates",
                       "crossover_policy_states", "crossover_skips",
                       "generation_summary"),
            normalizer=archive.normalizer, **kwargs)
        if self.model_choice_probability < 1.0:
            self.decision_counts["base_mixture_choice"] = 0
        trainer.observation_provider = self._observation

    def _event_uses_model(self, has_learnable_cut):
        if not has_learnable_cut:
            return False
        if self.model_choice_event is not None:
            return self.event_count == self.model_choice_event
        if self.model_choice_probability == 1.0:
            return True
        if self.model_choice_probability == 0.0:
            return False
        return self.control_rng.random() < self.model_choice_probability

    def evaluate(self, function, ind, *args, **kwargs):
        previous = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        is_pending_crossover = id(ind) in self._pending_child_ids
        if not is_pending_crossover:
            self.archive.observe(ind.objectives, bool(ind.feasible), fingerprint(ind))
        if (self.generation < 0 and self.evaluations > previous
                and len(self.current_population) < self.pop_size):
            self.current_population.append(ind)
        elif self.generation >= 0 and self.phase == "offspring" and self.evaluations > previous:
            self._record_pending_offspring(ind)
        return result

    def _record_pending_offspring(self, ind):
        key = fingerprint(ind)
        if key not in self._pending_fingerprints:
            self._pending_fingerprints.add(key)
            self.pending_offspring.append(ind)

    def _observation(self, candidates, context):
        if self._current_parents is None:
            raise RuntimeError("crossover observation requested outside a cut decision")
        parent1, parent2 = self._current_parents
        enriched = dict(context)
        enriched["offspring_fraction"] = (
            len(self.pending_offspring) / max(1, self.pop_size))
        if self._shared_observation is None:
            self._shared_observation = build_crossover_shared_observation(
                parent1, parent2, self.batches_for_observation, enriched,
                self.current_population, self.pending_offspring,
                self.archive.summary(), self.archive.normalizer, self.budget,
                self.evaluations, self.generations, self.graph_observation)
            self.shared_observation_builds += 1
        observation = build_crossover_observation(
            parent1, parent2, self.batches_for_observation, candidates, enriched,
            self.current_population, self.pending_offspring,
            self.archive.summary(), self.archive.normalizer, self.budget,
            self.evaluations, self.generations, self.graph_observation,
            shared_observation=self._shared_observation,
            candidate_representation=(
                getattr(getattr(self.trainer, "config", None),
                        "candidate_representation", "legacy")))
        self._last_observation = observation
        return observation

    def crossover_common_node(self, parent1, parent2, batches, tt_dict, arc_lookup):
        if self._active_event is not None:
            raise RuntimeError("previous crossover event was not settled")
        started = time.perf_counter()
        child1, child2, cut_decisions = prepare_common_node_crossover(
            parent1, parent2, batches, tt_dict=tt_dict)
        event_id = f"{self.run_id}:C:{self.event_count + 1}"
        self.event_count += 1
        self.total_attempts += 1
        self._shared_observation = None
        self._last_observation = None
        has_learnable_cut = any(
            len(cut.candidates) >= 2 for cut in cut_decisions)
        representation = getattr(
            getattr(self.trainer, "config", None),
            "candidate_representation", "legacy")
        if representation == "path_v3":
            return self._crossover_joint_override(
                parent1, parent2, child1, child2, cut_decisions, event_id,
                has_learnable_cut, started, tt_dict, arc_lookup)
        event_uses_model = self._event_uses_model(has_learnable_cut)
        # One crossover is one environment action and one joint reward.  The
        # mixture branch is therefore sampled once for the whole event, not
        # independently for its sub-decisions.
        use_model = [
            bool(event_uses_model and len(cut.candidates) >= 2)
            for cut in cut_decisions
        ]
        if has_learnable_cut:
            self.model_control_events += int(event_uses_model)
            self.base_control_events += int(not event_uses_model)
        model_decisions = sum(use_model)
        if model_decisions:
            self.trainer.begin_sequence()
        self._current_parents = (parent1, parent2)
        selected_rows = []
        changed = False
        try:
            for cut_index, cut in enumerate(cut_decisions):
                # The planner consumed Base's common-node draw immediately
                # after this Batch's allocation draws, preserving Base's
                # interleaved RNG order across multiple batches.
                shadow_base_index = cut.shadow_base_index
                context = {
                    "component": "C", "operator": "common_node",
                    "generation": self.generation, "phase": self.phase,
                    "K": self.instance.get("K"),
                    "deadline_window_alpha": self.instance.get(
                        "deadline_window_alpha"),
                }
                self._current_candidates = cut.candidates
                if len(cut.candidates) == 1:
                    # A forced cut is execution logic.  In particular, it must
                    # not consume or overwrite an earlier learnable cut that is
                    # waiting for this crossover's terminal reward.
                    decision = Decision(
                        0, [1.0], [None], {
                            "no_eligible_candidate": False,
                            "fallback": None,
                            "decision_kind": "single_candidate",
                        })
                elif use_model[cut_index]:
                    if self.trainer.pending is not None:
                        # Only the final learnable cut receives the terminal reward.
                        self.trainer.observe_outcome(0.0, done=False)
                    decision = self.trainer.choose(list(cut.candidates), context)
                    decision.metadata.update({
                        "control_branch": "model",
                        "model_choice_probability": self.model_choice_probability,
                    })
                else:
                    probabilities = [0.0] * len(cut.candidates)
                    probabilities[shadow_base_index] = 1.0
                    decision = Decision(
                        shadow_base_index, probabilities,
                        [None] * len(cut.candidates), {
                            "no_eligible_candidate": False,
                            "fallback": None,
                            "decision_kind": "base_mixture_choice",
                            "control_branch": "paired_base",
                            "model_choice_probability": (
                                self.model_choice_probability),
                        })
                kind = decision.metadata.get("decision_kind", "policy_choice")
                self.decision_counts[kind] = self.decision_counts.get(kind, 0) + 1
                self.inference_seconds += float(
                    decision.metadata.get("inference_seconds", 0.0))
                if decision.chosen_index is None:
                    self.no_candidate += 1
                    self.append("crossover_skips", {
                        "event_id": event_id, "cut_index": cut_index,
                        "batch_id": cut.batch.batch_id,
                        "reason": "no_eligible_candidate",
                        "decision_kind": kind,
                    })
                    continue
                chosen = cut.candidates[decision.chosen_index]
                changed = apply_cut(
                    child1, child2, cut, chosen, tt_dict, arc_lookup) or changed
                for candidate_index, candidate in enumerate(cut.candidates):
                    self.append("crossover_candidates", {
                        "event_id": event_id, "cut_index": cut_index,
                        "candidate_id": candidate_index,
                        "instance_id": self.instance.get("instance_id"),
                        "generation": self.generation,
                        "evaluation": self.evaluations,
                        "decision_kind": kind,
                        **candidate,
                        "baseline_selection_probability": (
                            1.0 / len(cut.candidates)),
                        "selection_probability": decision.probabilities[candidate_index],
                        "chosen": candidate_index == decision.chosen_index,
                    })
                selected_rows.append({
                    "cut_index": cut_index, "batch_id": cut.batch.batch_id,
                    "candidate_count": len(cut.candidates),
                    "chosen_index": decision.chosen_index,
                    "join_node": chosen["join_node"], "decision_kind": kind,
                    "shadow_base_index": shadow_base_index,
                    "different_from_shadow_base": (
                        decision.chosen_index != shadow_base_index),
                    "policy_metadata": decision.metadata,
                })
        except Exception:
            if model_decisions and self.trainer.sequence_depth:
                self.trainer.sequence_depth = 0
                self.trainer.sequence_policy_actions = 0
                self.trainer.pending = None
            raise
        finally:
            self._current_parents = None
            self._current_candidates = None
            self._shared_observation = None

        finalize_children(child1, child2)

        self.total_effective += int(changed)
        self._pending_child_ids = {id(child1), id(child2)}
        self._active_event = {
            "event_id": event_id,
            "crossover_event_index": self.event_count,
            "model_choice_event": self.model_choice_event,
            "generation": self.generation,
            "children": (child1, child2),
            "model_decisions": model_decisions,
            "cut_decision_count": len(cut_decisions),
            "selected_cuts": selected_rows,
            "changed_by_crossover": bool(changed),
            "started": started,
        }
        if not cut_decisions:
            self.no_candidate += 1
            self.decision_counts["no_candidate"] += 1
            self.append("crossover_skips", {
                "event_id": event_id, "instance_id": self.instance.get("instance_id"),
                "generation": self.generation, "evaluation": self.evaluations,
                "reason": "no_legal_common_node", "decision_kind": "no_candidate",
            })
        return child1, child2

    @staticmethod
    def _joint_override_candidates(cut_decisions):
        """Expose Base passthrough plus every single-cut deviation from Base."""
        overrides = []
        learnable = [
            (cut_index, cut)
            for cut_index, cut in enumerate(cut_decisions)
            if len(cut.candidates) >= 2
        ]
        cut_denominator = max(1, len(cut_decisions) - 1)
        largest_candidate_count = max(
            (len(cut.candidates) for _, cut in learnable), default=1)
        for cut_index, cut in learnable:
            for candidate_index, source in enumerate(cut.candidates):
                if candidate_index == cut.shadow_base_index:
                    continue
                candidate = dict(source)
                candidate.update({
                    "joint_action": "single_cut_override",
                    "action_group": "override",
                    "safe_action": False,
                    "joint_cut_index": cut_index,
                    "joint_candidate_index": candidate_index,
                    "is_base_passthrough": 0.0,
                    "cut_index_fraction": cut_index / cut_denominator,
                    "cut_candidate_count_fraction": (
                        len(cut.candidates) / largest_candidate_count),
                    "is_shadow_base": 0.0,
                })
                overrides.append(candidate)
        if not overrides:
            return []

        # The dedicated flag makes the safe action identifiable.  Retaining a
        # real shadow candidate supplies well-formed numerical/path tokens;
        # execution still applies every Base shadow cut, not just this row.
        _, first_cut = learnable[0]
        passthrough = dict(first_cut.candidates[first_cut.shadow_base_index])
        passthrough.update({
            "joint_action": "base_passthrough",
            "action_group": "base",
            "safe_action": True,
            "joint_cut_index": None,
            "joint_candidate_index": None,
            "is_base_passthrough": 1.0,
            "cut_index_fraction": 0.0,
            "cut_candidate_count_fraction": (
                len(first_cut.candidates) / largest_candidate_count),
            "is_shadow_base": 1.0,
        })
        override_prior = 0.5 / len(overrides)
        passthrough["selection_probability"] = 0.5
        for candidate in overrides:
            candidate["selection_probability"] = override_prior
        return [passthrough, *overrides]

    def _crossover_joint_override(
            self, parent1, parent2, child1, child2, cut_decisions, event_id,
            has_learnable_cut, started, tt_dict, arc_lookup):
        """Let one PPO action replace at most one cut in the paired Base plan."""
        joint_candidates = self._joint_override_candidates(cut_decisions)
        event_uses_model = self._event_uses_model(
            bool(joint_candidates) and has_learnable_cut)
        if has_learnable_cut:
            self.model_control_events += int(event_uses_model)
            self.base_control_events += int(not event_uses_model)

        decision = None
        selected_override = None
        if event_uses_model:
            self._current_parents = (parent1, parent2)
            self._current_candidates = joint_candidates
            self.trainer.begin_sequence()
            context = {
                "component": "C", "operator": "common_node_joint_override",
                "generation": self.generation, "phase": self.phase,
                "K": self.instance.get("K"),
                "deadline_window_alpha": self.instance.get(
                    "deadline_window_alpha"),
            }
            try:
                decision = self.trainer.choose(joint_candidates, context)
                if self.forced_joint_candidate_index is not None:
                    forced = self.forced_joint_candidate_index
                    if forced >= len(joint_candidates):
                        raise ValueError(
                            f"forced C joint candidate index {forced} is out of "
                            f"range for {len(joint_candidates)} candidates")
                    decision.chosen_index = forced
                    decision.metadata.update({
                        "forced_joint_candidate_index": forced,
                        "diagnostic_forced_action": True,
                        "selected_action_group": joint_candidates[forced].get(
                            "action_group"),
                    })
                    forced_probability = float(decision.probabilities[forced])
                    if forced_probability > 0.0:
                        forced_log_probability = math.log(forced_probability)
                        decision.metadata["old_log_probability"] = (
                            forced_log_probability)
                        pending = getattr(self.trainer, "pending", None)
                        if isinstance(pending, dict):
                            pending["action"] = forced
                            pending["old_log_probability"] = (
                                forced_log_probability)
                decision.metadata.update({
                    "control_branch": "model",
                    "model_choice_probability": self.model_choice_probability,
                    "crossover_control": (
                        "hierarchical-single-joint-override-with-base-passthrough"),
                })
                chosen = joint_candidates[decision.chosen_index]
                if chosen["joint_action"] == "single_cut_override":
                    selected_override = (
                        int(chosen["joint_cut_index"]),
                        int(chosen["joint_candidate_index"]),
                    )
                kind = decision.metadata.get("decision_kind", "model_choice")
                if self._last_observation is None:
                    raise RuntimeError("C policy observation was not captured")
                self.append("crossover_policy_states", {
                    "event_id": event_id,
                    "crossover_event_index": self.event_count,
                    "instance_id": self.instance.get("instance_id"),
                    "generation": self.generation,
                    "evaluation": self.evaluations,
                    "candidate_count": len(joint_candidates),
                    "chosen_index": decision.chosen_index,
                    "probabilities": decision.probabilities,
                    "actions": [{
                        "joint_action": candidate.get("joint_action"),
                        "action_group": candidate.get("action_group"),
                        "safe_action": candidate.get("safe_action"),
                        "joint_cut_index": candidate.get("joint_cut_index"),
                        "joint_candidate_index": candidate.get(
                            "joint_candidate_index"),
                        "batch_id": candidate.get("batch_id"),
                        "join_node": candidate.get("join_node"),
                    } for candidate in joint_candidates],
                    "observation": self._last_observation,
                })
                self.decision_counts[kind] = self.decision_counts.get(kind, 0) + 1
                self.inference_seconds += float(
                    decision.metadata.get("inference_seconds", 0.0))
                for candidate_index, candidate in enumerate(joint_candidates):
                    self.append("crossover_candidates", {
                        "event_id": event_id,
                        "cut_index": candidate.get("joint_cut_index"),
                        "candidate_id": candidate_index,
                        "instance_id": self.instance.get("instance_id"),
                        "generation": self.generation,
                        "evaluation": self.evaluations,
                        "decision_kind": kind,
                        **candidate,
                        "baseline_selection_probability": candidate[
                            "selection_probability"],
                        "selection_probability": decision.probabilities[
                            candidate_index],
                        "chosen": candidate_index == decision.chosen_index,
                    })
            except Exception:
                self.trainer.sequence_depth = 0
                self.trainer.sequence_policy_actions = 0
                self.trainer.pending = None
                raise
            finally:
                self._current_parents = None
                self._current_candidates = None
                self._shared_observation = None
                self._last_observation = None

        reference_override = None
        reference_action = None
        if (decision is not None
                and self.crossover_reward_mode in (
                    "paired_base_relative", "paired_archive_then_quality")):
            if selected_override is None:
                override_indices = [
                    index for index, candidate in enumerate(joint_candidates)
                    if candidate["joint_action"] == "single_cut_override"
                ]
                reference_index = max(
                    override_indices,
                    key=lambda index: decision.probabilities[index])
                reference_candidate = joint_candidates[reference_index]
                reference_override = (
                    int(reference_candidate["joint_cut_index"]),
                    int(reference_candidate["joint_candidate_index"]),
                )
                reference_action = "single_cut_override"
            else:
                reference_action = "base_passthrough"

        reference_children = None
        if reference_action is not None:
            reference_children = (deepcopy(child1), deepcopy(child2))

        selected_rows = []
        changed = False
        for cut_index, cut in enumerate(cut_decisions):
            candidate_index = (
                selected_override[1]
                if selected_override is not None
                and selected_override[0] == cut_index else
                cut.shadow_base_index)
            candidate = cut.candidates[candidate_index]
            changed = apply_cut(
                child1, child2, cut, candidate, tt_dict, arc_lookup) or changed
            selected_rows.append({
                "cut_index": cut_index,
                "batch_id": cut.batch.batch_id,
                "candidate_count": len(cut.candidates),
                "chosen_index": candidate_index,
                "join_node": candidate["join_node"],
                "decision_kind": (
                    "model_single_cut_override"
                    if selected_override is not None
                    and selected_override[0] == cut_index else
                    "paired_base_cut"),
                "shadow_base_index": cut.shadow_base_index,
                "different_from_shadow_base": (
                    candidate_index != cut.shadow_base_index),
                "policy_metadata": (
                    decision.metadata if decision is not None else None),
            })

            if reference_children is not None:
                reference_index = (
                    reference_override[1]
                    if reference_override is not None
                    and reference_override[0] == cut_index else
                    cut.shadow_base_index)
                apply_cut(
                    reference_children[0], reference_children[1], cut,
                    cut.candidates[reference_index], tt_dict, arc_lookup)

        finalize_children(child1, child2)
        if reference_children is not None:
            finalize_children(*reference_children)
        self.total_effective += int(changed)
        self._pending_child_ids = {id(child1), id(child2)}
        self._active_event = {
            "event_id": event_id,
            "crossover_event_index": self.event_count,
            "model_choice_event": self.model_choice_event,
            "forced_joint_candidate_index": self.forced_joint_candidate_index,
            "joint_candidate_count": len(joint_candidates),
            "generation": self.generation,
            "children": (child1, child2),
            "model_decisions": int(event_uses_model),
            "cut_decision_count": len(cut_decisions),
            "selected_cuts": selected_rows,
            "changed_by_crossover": bool(changed),
            "crossover_control": (
                "hierarchical-single-joint-override-with-base-passthrough"),
            "selected_joint_action": (
                None if decision is None else
                joint_candidates[decision.chosen_index]["joint_action"]),
            "selected_joint_candidate_index": (
                None if decision is None else int(decision.chosen_index)),
            "policy_metadata": (
                None if decision is None else dict(decision.metadata)),
            "paired_reference_action": reference_action,
            "paired_reference_children": reference_children,
            "started": started,
        }
        if not cut_decisions:
            self.no_candidate += 1
            self.decision_counts["no_candidate"] += 1
            self.append("crossover_skips", {
                "event_id": event_id,
                "instance_id": self.instance.get("instance_id"),
                "generation": self.generation,
                "evaluation": self.evaluations,
                "reason": "no_legal_common_node",
                "decision_kind": "no_candidate",
            })
        return child1, child2

    def evaluate_crossover_counterfactual(
            self, function, child1, child2, *args, **kwargs):
        """Evaluate selected/reference pre-mutation children outside search budget."""
        event = self._active_event
        if (event is None or self.crossover_reward_mode not in (
                "paired_base_relative", "paired_archive_then_quality")
                or not event.get("model_decisions")):
            return
        reference_children = event.get("paired_reference_children")
        if reference_children is None:
            raise RuntimeError("paired crossover reward is missing its reference action")

        selected = (deepcopy(child1), deepcopy(child2))
        reference = tuple(deepcopy(item) for item in reference_children)
        started = time.perf_counter()
        for individual in (*selected, *reference):
            function(individual, *args, **kwargs)
        self.counterfactual_evaluation_seconds += time.perf_counter() - started
        self.counterfactual_evaluations += 4

        selected_candidates = [
            (item.objectives, bool(item.feasible), fingerprint(item))
            for item in selected
        ]
        reference_candidates = [
            (item.objectives, bool(item.feasible), fingerprint(item))
            for item in reference
        ]
        selected_update = self.archive.preview_many(selected_candidates)
        reference_update = self.archive.preview_many(reference_candidates)
        archive_relative_delta = (
            selected_update.delta_hv - reference_update.delta_hv)
        selected_quality_hv = self.archive.standalone_hv(selected_candidates)
        reference_quality_hv = self.archive.standalone_hv(reference_candidates)
        quality_relative_delta = selected_quality_hv - reference_quality_hv
        if self.crossover_reward_mode == "paired_archive_then_quality":
            if abs(archive_relative_delta) > self.reward_config.floating_tolerance:
                relative_delta = archive_relative_delta
                label_source = "archive"
            elif abs(quality_relative_delta) > self.reward_config.floating_tolerance:
                relative_delta = quality_relative_delta
                label_source = "standalone_quality"
            else:
                relative_delta = 0.0
                label_source = "tie"
        else:
            relative_delta = archive_relative_delta
            label_source = "archive" if abs(relative_delta) > (
                self.reward_config.floating_tolerance) else "tie"
        event["paired_reward"] = signed_hv_reward(
            relative_delta, self.reward_config)
        event["paired_selected_hv_delta"] = float(
            selected_update.delta_hv)
        event["paired_reference_hv_delta"] = float(
            reference_update.delta_hv)
        event["paired_archive_relative_hv"] = float(
            archive_relative_delta)
        event["paired_selected_standalone_hv"] = float(
            selected_quality_hv)
        event["paired_reference_standalone_hv"] = float(
            reference_quality_hv)
        event["paired_standalone_quality_relative_hv"] = float(
            quality_relative_delta)
        event["paired_reward_label_source"] = label_source
        event["paired_selected_minus_reference_hv"] = float(relative_delta)
        event["paired_counterfactual_evaluations"] = 4
        event["paired_reference_children"] = None

    def settle_crossover(self, child1, child2):
        event = self._active_event
        if event is None:
            return
        if {id(child1), id(child2)} != self._pending_child_ids:
            raise RuntimeError("final offspring do not match the pending crossover event")
        child_fingerprints = (fingerprint(child1), fingerprint(child2))
        reward_diagnostics = [
            self.archive.diagnose_candidate(
                child.objectives, bool(child.feasible), child_fingerprint)
            for child, child_fingerprint in zip(
                (child1, child2), child_fingerprints)
        ]
        update = self.archive.observe_many([
            (child1.objectives, bool(child1.feasible), child_fingerprints[0]),
            (child2.objectives, bool(child2.feasible), child_fingerprints[1]),
        ])
        archive_reward = hv_reward(update.delta_hv, self.reward_config)
        if (self.crossover_reward_mode in (
                "paired_base_relative", "paired_archive_then_quality")
                and event["model_decisions"]):
            if "paired_reward" not in event:
                raise RuntimeError(
                    "paired crossover reward was not evaluated before mutation")
            reward = float(event["paired_reward"])
        else:
            reward = archive_reward
        joint_action = event.get("selected_joint_action")
        if joint_action in self.joint_action_reward_stats:
            stats = self.joint_action_reward_stats[joint_action]
            base_stats = self.joint_action_reward_stats["base_passthrough"]
            base_mean_before = (
                None if base_stats["operations"] == 0 else
                base_stats["reward_total"] / base_stats["operations"])
            event["prior_base_reward_mean"] = base_mean_before
            event["reward_minus_prior_base_mean"] = (
                None if base_mean_before is None else reward - base_mean_before)
            stats["operations"] += 1
            stats["positive"] += int(reward > 0.0)
            stats["reward_total"] += float(reward)
            stats["raw_hv_total"] += float(event.get(
                "paired_selected_minus_reference_hv", update.delta_hv))
        if event["model_decisions"]:
            self.trainer.end_sequence(
                reward, done=self.credit_assignment == "operation_terminal")
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0.0)
        event.update({
            "reward": float(reward), "reward_hv_delta": float(update.delta_hv),
            "archive_reward": float(archive_reward),
            "crossover_reward_mode": self.crossover_reward_mode,
            "reward_diagnostics": reward_diagnostics,
            "reward_positive_hv": bool(update.delta_hv > 0.0),
            "reward_archive_changed_without_hv": bool(
                any(item["would_change_archive"] for item in reward_diagnostics)
                and update.delta_hv <= 0.0),
            "reward_archive_hv_before": update.hv_before,
            "reward_archive_hv_after": update.hv_after,
            "reward_archive_size": update.archive_size,
            "reward_recorded_for_policy": bool(event["model_decisions"]),
            "evaluation": self.evaluations,
            "child_1": child1, "child_2": child2,
            "child_1_fingerprint": child_fingerprints[0],
            "child_2_fingerprint": child_fingerprints[1],
            "child_1_feasible": bool(child1.feasible),
            "child_2_feasible": bool(child2.feasible),
            "action_wall_seconds": time.perf_counter() - event["started"],
        })
        event.pop("children", None)
        event.pop("paired_reference_children", None)
        event.pop("started", None)
        self._settled_outcomes.append(event)
        if self.phase == "offspring":
            self._record_pending_offspring(child1)
            self._record_pending_offspring(child2)
        self._pending_child_ids.clear()
        self._active_event = None

    def flush_events(self, population):
        for row in self._settled_outcomes:
            child1, child2 = row.pop("child_1"), row.pop("child_2")
            row["child_1_survived"] = any(child1 is item for item in population)
            row["child_2_survived"] = any(child2 is item for item in population)
            self.append("crossover_outcomes", row)
        self._settled_outcomes.clear()
        self.current_population = list(population)
        self.pending_offspring.clear()
        self._pending_fingerprints.clear()

    def log_generation(self, population, boost):
        self.generations_completed += 1
        now = time.perf_counter()
        self.append("generation_summary", {
            "run_id": self.run_id, "generation": self.generation,
            "real_objective_evaluation_count": self.evaluations,
            "crossover_attempts_cumulative": self.total_attempts,
            "effective_crossover_rate_cumulative": (
                self.total_effective / max(1, self.total_attempts)),
            "feasible_ratio": sum(item.feasible for item in population) / len(population),
            "boost_triggered": bool(boost), "runtime_seconds": now - self.start,
            "generation_seconds": now - self._last_generation_time,
            "inference_seconds_cumulative": self.inference_seconds,
            "no_eligible_candidate_cumulative": self.no_candidate,
            "decision_counts_cumulative": dict(self.decision_counts),
        })
        self._last_generation_time = now
        self.log_search_progress(population, "environmental_selection")

    def finish_episode(self):
        if self._active_event is not None:
            raise RuntimeError("episode ended before a crossover reward was settled")
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
            "credit_assignment": self.credit_assignment,
            "crossover_reward_mode": self.crossover_reward_mode,
            "counterfactual_evaluation_count": self.counterfactual_evaluations,
            "counterfactual_evaluation_seconds": (
                self.counterfactual_evaluation_seconds),
            "crossover_control": (
                "hierarchical-single-joint-override-with-base-passthrough"
                if getattr(getattr(self.trainer, "config", None),
                           "candidate_representation", "legacy") == "path_v3"
                else "per-cut"),
            "shared_observation_builds": self.shared_observation_builds,
            "model_choice_probability": self.model_choice_probability,
            "model_control_events": self.model_control_events,
            "base_control_events": self.base_control_events,
            "joint_action_reward_stats": self.joint_action_reward_stats,
            "feature_construction_seconds": self.feature_construction_seconds,
            "ppo_update_seconds": self.ppo_update_seconds,
        }
