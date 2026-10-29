"""PPO controller for sequential distinct-path initialization."""
from __future__ import annotations

import random
import time

import baseline_uncertainty as base
from multi_instance_logging import MutationLogger, fingerprint
from multi_instance_policy import Decision

from .archive import RewardArchive, RewardConfig, hv_reward
from .initialization_observations import build_initialization_observation


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


class InitializationPPOController(MutationLogger):
    """Change only Base's random path identities during initialization."""

    controls_mutation = False
    controls_crossover = False
    controls_initialization = True

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
        self._current_partial = None
        self._current_context = None
        self._active_event = None
        self.reward_total = 0.0
        self.positive_reward_actions = 0
        super().__init__(
            *args, policy=trainer, evaluate_before_mutation=False,
            log_names=("initialization_outcomes", "initialization_candidates",
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
            self.evaluations, self.pop_size, self.graph_observation)

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

                k = random.randint(1, min(int(max_paths), len(paths)))
                # Consume Base's exact path-sampling stream even though the
                # policy replaces the result.  All other stochastic choices,
                # including k and split shares, therefore remain paired.
                shadow_paths = random.sample(paths, k)
                raw_shares = [random.random() for _ in range(k)]
                remaining = list(paths)
                selected = []
                for step in range(k):
                    candidates = [{
                        "eligible": True, "path": path,
                        "path_id": path.path_id,
                        "nodes": list(path.nodes), "modes": list(path.modes),
                    } for path in remaining]
                    context = {
                        "component": "I", "batch": batch,
                        "selection_step": step,
                        "required_path_count": k,
                        "K": self.instance.get("K"),
                        "deadline_window_alpha": self.instance.get(
                            "deadline_window_alpha"),
                    }
                    self._current_partial = individual
                    self._current_context = context
                    if len(candidates) == 1:
                        decision = Decision(
                            0, [1.0], [None], {
                                "no_eligible_candidate": False,
                                "fallback": None,
                                "decision_kind": "single_candidate",
                            })
                    else:
                        if not sequence_started:
                            self.trainer.begin_sequence()
                            sequence_started = True
                        elif self.trainer.pending is not None:
                            self.trainer.observe_outcome(0.0, done=False)
                        decision = self.trainer.choose(candidates, context)
                        model_decisions += 1
                    kind = decision.metadata["decision_kind"]
                    self.decision_counts[kind] = self.decision_counts.get(kind, 0) + 1
                    self.inference_seconds += float(
                        decision.metadata.get("inference_seconds", 0.0))
                    chosen_index = decision.chosen_index
                    if chosen_index is None:
                        raise RuntimeError("non-empty initialization candidates were rejected")
                    chosen = remaining.pop(chosen_index)
                    selected.append(chosen)
                    individual.od_allocations[key] = [
                        base.PathAllocation(path=path, share=raw_shares[index])
                        for index, path in enumerate(selected)
                    ]
                    for candidate_index, candidate in enumerate(candidates):
                        self.append("initialization_candidates", {
                            "event_id": event_id, "batch_id": batch.batch_id,
                            "selection_step": step, "required_path_count": k,
                            "candidate_id": candidate_index,
                            "path_id": candidate["path_id"],
                            "nodes": candidate["nodes"], "modes": candidate["modes"],
                            "decision_kind": kind,
                            "selection_probability": decision.probabilities[candidate_index],
                            "chosen": candidate_index == chosen_index,
                        })
                    selected_rows.append({
                        "batch_id": batch.batch_id, "selection_step": step,
                        "required_path_count": k, "chosen_path_id": chosen.path_id,
                        "decision_kind": kind,
                    })
                individual.od_allocations[key] = base.merge_and_normalize(
                    individual.od_allocations[key])
                # Order matters because Base attaches the independently drawn
                # split shares to sampled paths in order.
                differs_from_shadow = differs_from_shadow or (
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
            "feature_construction_seconds": self.feature_construction_seconds,
            "ppo_update_seconds": self.ppo_update_seconds,
        }
