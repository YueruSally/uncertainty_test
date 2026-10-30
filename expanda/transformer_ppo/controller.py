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
from multi_instance_mutation import candidates_for

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
            "mutation_policy_states", "generation_summary",
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
                or self.model_choice_event is not None):
            self.decision_counts["base_mixture_choice"] = 0
        trainer.observation_provider = self._observation

    def _next_control_branch(self):
        """Return the branch and globally ordered mutation-call index."""
        self.mixture_event_count += 1
        if self.model_choice_event is not None:
            use_model = self.mixture_event_count == self.model_choice_event
        else:
            use_model = (
                True if self.model_choice_probability == 1.0 else
                False if self.model_choice_probability == 0.0 else
                self.control_rng.random() < self.model_choice_probability
            )
        return use_model, self.mixture_event_count

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
                "mixture_event_index": mixture_event_index,
                "operator": result[0],
                "decision_changed": bool(changed),
                "reward_recorded_for_policy": False,
            })
            if self.phase == "offspring":
                self._record_pending_offspring(ind)
            return result

        self.model_control_events += 1
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
