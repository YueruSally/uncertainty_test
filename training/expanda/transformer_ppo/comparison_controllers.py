"""Budget-only Base and frozen I+C+M coordination controllers."""
from __future__ import annotations

import time

from multi_instance_logging import MutationLogger, fingerprint, write_json

from .archive import RewardArchive, RewardConfig
from .controller import PPOController
from .crossover_controller import CrossoverPPOController
from .initialization_controller import InitializationPPOController


class BaseBudgetController(MutationLogger):
    """Count evaluations without changing any original NSGA-II component."""

    controls_initialization = False
    controls_crossover = False
    controls_mutation = False

    def __init__(self, *args, normalizer=None, **kwargs):
        self.archive = RewardArchive(normalizer) if normalizer is not None else None
        self._archive_pending = []
        super().__init__(
            *args, evaluate_before_mutation=False,
            log_names=("generation_summary",), normalizer=normalizer, **kwargs)

    def evaluate(self, function, ind, *args, **kwargs):
        previous = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        if self.archive is not None and self.evaluations > previous:
            # Base has no policy reward to settle after each solution.  Delay
            # the read-only archive update until the next logged selection
            # boundary, where the visible set and HV are exactly the same as
            # under sequential insertion but avoid hundreds of intermediate
            # exact-HV calculations.
            self._archive_pending.append(
                (ind.objectives, bool(ind.feasible), fingerprint(ind)))
        return result

    def _flush_archive(self):
        if self.archive is not None and self._archive_pending:
            self.archive.observe_many(self._archive_pending)
            self._archive_pending.clear()

    def log_initial_population(self, population):
        self._flush_archive()
        write_json(self.out / "initial_feasible_objectives.json", [{
            "decision_fingerprint": fingerprint(individual),
            "objectives": [float(value) for value in individual.objectives],
        } for individual in population if individual.feasible])
        super().log_initial_population(population)

    def flush_events(self, population):
        return None

    def log_generation(self, population, boost):
        self._flush_archive()
        self.generations_completed += 1
        now = time.perf_counter()
        self.append("generation_summary", {
            "run_id": self.run_id,
            "generation": self.generation,
            "real_objective_evaluation_count": self.evaluations,
            "feasible_ratio": sum(item.feasible for item in population) / len(population),
            "boost_triggered": bool(boost),
            "runtime_seconds": now - self.start,
            "generation_seconds": now - self._last_generation_time,
        })
        self._last_generation_time = now
        self.log_search_progress(population, "environmental_selection")

    def finish_episode(self):
        self._flush_archive()

    def summary(self):
        return {
            "reward_total": 0.0,
            "positive_reward_actions": 0,
            "decision_counts": dict(self.decision_counts),
            "ppo_updates": 0,
            "ppo_updates_before_run": 0,
            "ppo_updates_total": 0,
            "feature_construction_seconds": 0.0,
            "ppo_update_seconds": 0.0,
        }


class CombinedPPOController:
    """Route one NSGA-II run through three separately frozen policies."""

    controls_initialization = True
    controls_crossover = True
    controls_mutation = True

    def __init__(self, out, run_id, scenario_id, budget, *, trainers,
                 normalizer, reward_config: RewardConfig, batches, generations,
                 pop_size, graph_observation, method, instance, seeds):
        if set(trainers) != {"I", "C", "M"}:
            raise ValueError("combined controller requires exactly I, C, and M trainers")
        if any(trainer.training for trainer in trainers.values()):
            raise ValueError("combined policies must be frozen before composition")
        self.out = out
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.budget = budget
        self.method = method
        self.instance = instance
        self.seeds = seeds
        self.evaluations = 0
        self.cache_hits = 0
        self.generations_completed = 0
        self._generation = -1
        self._phase = "initialisation"
        self.start = time.perf_counter()
        self.objective_evaluation_seconds = 0.0

        controller_types = {
            "I": InitializationPPOController,
            "C": CrossoverPPOController,
            "M": PPOController,
        }
        self.controllers = {}
        for component in ("I", "C", "M"):
            component_out = out / "components" / component
            component_out.mkdir(parents=True, exist_ok=False)
            self.controllers[component] = controller_types[component](
                component_out, run_id, scenario_id, budget,
                trainer=trainers[component], archive=RewardArchive(normalizer),
                reward_config=reward_config, batches=batches,
                generations=generations, pop_size=pop_size,
                graph_observation=graph_observation, method=method,
                instance=instance, seeds=seeds)
        self._sync_children()

    @property
    def generation(self):
        return self._generation

    @generation.setter
    def generation(self, value):
        self._generation = int(value)
        if hasattr(self, "controllers"):
            self._sync_children()

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = str(value)
        if hasattr(self, "controllers"):
            self._sync_children()

    @property
    def inference_seconds(self):
        return sum(item.inference_seconds for item in self.controllers.values())

    @property
    def feature_construction_seconds(self):
        return sum(item.feature_construction_seconds
                   for item in self.controllers.values())

    @property
    def ppo_update_seconds(self):
        return sum(item.ppo_update_seconds for item in self.controllers.values())

    def _sync_children(self):
        for controller in self.controllers.values():
            controller.generation = self._generation
            controller.phase = self._phase
            controller.evaluations = self.evaluations
            controller.cache_hits = self.cache_hits
            controller.objective_evaluation_seconds = (
                self.objective_evaluation_seconds)

    def has_budget(self, required):
        return self.budget is None or self.evaluations + int(required) <= self.budget

    def budget_reserve(self, stage, units=1):
        units = int(units)
        if units < 1:
            raise ValueError("budget reserve units must be positive")
        if stage == "offspring_pair":
            return 2 * units
        if stage == "feasibility_boost":
            return units + (units % 2)
        raise ValueError(f"unknown budget reserve stage: {stage}")

    def evaluate(self, function, ind, *args, **kwargs):
        key = (self.scenario_id, fingerprint(ind))
        if getattr(ind, "_learning_eval_key", None) == key:
            self.cache_hits += 1
            self._sync_children()
            return
        if not self.has_budget(1):
            raise RuntimeError("objective evaluation budget exhausted")
        evaluation_started = time.perf_counter()
        function(ind, *args, **kwargs)
        self.objective_evaluation_seconds += time.perf_counter() - evaluation_started
        self.evaluations += 1
        ind._learning_eval_key = key
        ind._learning_eval_id = self.evaluations
        self._sync_children()
        self._observe_evaluated(ind)

    def _observe_evaluated(self, ind):
        identity = fingerprint(ind)
        initial = self.controllers["I"]
        crossover = self.controllers["C"]
        mutation = self.controllers["M"]

        active_initial = (
            initial._active_event is not None
            and id(ind) == initial._active_event["individual_id"])
        if not active_initial:
            initial.archive.observe(ind.objectives, bool(ind.feasible), identity)
        if id(ind) not in crossover._pending_child_ids:
            crossover.archive.observe(ind.objectives, bool(ind.feasible), identity)
        if not mutation._suppress_archive:
            mutation.archive.observe(ind.objectives, bool(ind.feasible), identity)

        if self.generation < 0:
            initial.current_population.append(ind)
            if len(crossover.current_population) < crossover.pop_size:
                crossover.current_population.append(ind)
            if len(mutation.current_population) < mutation.pop_size:
                mutation.current_population.append(ind)
        elif self.phase == "offspring":
            crossover._record_pending_offspring(ind)
            if not mutation._inside_mutation:
                mutation._record_pending_offspring(ind)

    def initial_individual(self, *args, **kwargs):
        self._sync_children()
        return self.controllers["I"].initial_individual(*args, **kwargs)

    def settle_initialization(self, *args, **kwargs):
        self._sync_children()
        return self.controllers["I"].settle_initialization(*args, **kwargs)

    def crossover_common_node(self, *args, **kwargs):
        self._sync_children()
        return self.controllers["C"].crossover_common_node(*args, **kwargs)

    def settle_crossover(self, *args, **kwargs):
        self._sync_children()
        return self.controllers["C"].settle_crossover(*args, **kwargs)

    def mutate(self, *args, **kwargs):
        self._sync_children()
        result = self.controllers["M"].mutate(*args, **kwargs)
        self._sync_children()
        return result

    def flush_events(self, population):
        self._sync_children()
        for controller in self.controllers.values():
            controller.flush_events(population)

    def log_generation(self, population, boost):
        self._sync_children()
        for controller in self.controllers.values():
            controller.log_generation(population, boost)
        self.generations_completed = max(
            controller.generations_completed
            for controller in self.controllers.values())

    def log_initial_population(self, population):
        self._sync_children()
        for controller in self.controllers.values():
            controller.log_initial_population(population)

    def finish_episode(self):
        self._sync_children()
        for controller in self.controllers.values():
            controller.finish_episode()

    def close(self):
        for controller in self.controllers.values():
            controller.close()

    def summary(self):
        summaries = {
            component: controller.summary()
            for component, controller in self.controllers.items()
        }
        return {
            "component_summaries": summaries,
            "reward_total": sum(row["reward_total"] for row in summaries.values()),
            "positive_reward_actions": sum(
                row["positive_reward_actions"] for row in summaries.values()),
            "decision_counts": {
                component: row["decision_counts"]
                for component, row in summaries.items()
            },
            "ppo_updates": {
                component: row["ppo_updates"] for component, row in summaries.items()
            },
            "ppo_updates_before_run": {
                component: row["ppo_updates_before_run"]
                for component, row in summaries.items()
            },
            "ppo_updates_total": {
                component: row["ppo_updates_total"]
                for component, row in summaries.items()
            },
        }
