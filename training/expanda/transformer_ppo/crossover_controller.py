"""PPO controller for common-node crossover-cut selection."""
from __future__ import annotations

import random
import time

from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_policy import Decision

from .archive import RewardArchive, RewardConfig, hv_reward
from .crossover import apply_cut, finalize_children, prepare_common_node_crossover
from .crossover_observations import (
    build_crossover_observation,
    build_crossover_shared_observation,
)


class CrossoverPPOController(MutationLogger):
    """Keep Base crossover structure and settle two final children jointly."""

    controls_mutation = False
    controls_crossover = True

    def __init__(self, *args, trainer, archive: RewardArchive,
                 reward_config: RewardConfig, batches, generations, pop_size,
                 graph_observation=None, **kwargs):
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
        self._current_parents = None
        self._current_candidates = None
        self._shared_observation = None
        self.shared_observation_builds = 0
        self._active_event = None
        self._pending_child_ids = set()
        self._settled_outcomes = []
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            log_names=("crossover_outcomes", "crossover_candidates",
                       "crossover_skips", "generation_summary"),
            normalizer=archive.normalizer, **kwargs)
        trainer.observation_provider = self._observation

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
        return build_crossover_observation(
            parent1, parent2, self.batches_for_observation, candidates, enriched,
            self.current_population, self.pending_offspring,
            self.archive.summary(), self.archive.normalizer, self.budget,
            self.evaluations, self.generations, self.graph_observation,
            shared_observation=self._shared_observation)

    def crossover_common_node(self, parent1, parent2, batches, tt_dict, arc_lookup):
        if self._active_event is not None:
            raise RuntimeError("previous crossover event was not settled")
        started = time.perf_counter()
        child1, child2, cut_decisions = prepare_common_node_crossover(
            parent1, parent2, batches)
        event_id = f"{self.run_id}:C:{self.event_count + 1}"
        self.event_count += 1
        self.total_attempts += 1
        self._shared_observation = None
        model_decisions = sum(len(item.candidates) >= 2 for item in cut_decisions)
        if model_decisions:
            self.trainer.begin_sequence()
        self._current_parents = (parent1, parent2)
        selected_rows = []
        changed = False
        try:
            for cut_index, cut in enumerate(cut_decisions):
                # Base would draw one common node here.  Consume the paired
                # draw before replacing its result with the frozen policy.
                shadow_base_index = random.randrange(len(cut.candidates))
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
                else:
                    if self.trainer.pending is not None:
                        # Only the final learnable cut receives the terminal reward.
                        self.trainer.observe_outcome(0.0, done=False)
                    decision = self.trainer.choose(list(cut.candidates), context)
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

    def settle_crossover(self, child1, child2):
        event = self._active_event
        if event is None:
            return
        if {id(child1), id(child2)} != self._pending_child_ids:
            raise RuntimeError("final offspring do not match the pending crossover event")
        update = self.archive.observe_many([
            (child1.objectives, bool(child1.feasible), fingerprint(child1)),
            (child2.objectives, bool(child2.feasible), fingerprint(child2)),
        ])
        reward = hv_reward(update.delta_hv, self.reward_config)
        if event["model_decisions"]:
            self.trainer.end_sequence(reward, done=False)
            self.reward_total += reward
            self.positive_reward_actions += int(reward > 0.0)
        event.update({
            "reward": float(reward), "reward_hv_delta": float(update.delta_hv),
            "reward_archive_hv_before": update.hv_before,
            "reward_archive_hv_after": update.hv_after,
            "reward_archive_size": update.archive_size,
            "reward_recorded_for_policy": bool(event["model_decisions"]),
            "evaluation": self.evaluations,
            "child_1": child1, "child_2": child2,
            "child_1_fingerprint": fingerprint(child1),
            "child_2_fingerprint": fingerprint(child2),
            "child_1_feasible": bool(child1.feasible),
            "child_2_feasible": bool(child2.feasible),
            "action_wall_seconds": time.perf_counter() - event["started"],
        })
        event.pop("children", None)
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
            "shared_observation_builds": self.shared_observation_builds,
            "feature_construction_seconds": self.feature_construction_seconds,
            "ppo_update_seconds": self.ppo_update_seconds,
        }
