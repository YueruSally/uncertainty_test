"""Frozen I+C+M K8 coordination for the common-seed component screen."""
from __future__ import annotations

from pathlib import Path

from crossover import FrozenCResidualGate
from initialization import FrozenIResidualGate
from runtime_common import digest, read_json
from mutation import ScaleSearchController
from multi_instance_logging import fingerprint
from transformer_ppo.archive import RewardArchive
from crossover import CrossoverK8Controller
from initialization import InitializationK8Controller


def create_component_directories(out):
    root = Path(out) / "components"
    root.mkdir(parents=True, exist_ok=False)
    for component in ("I", "C"):
        (root / component).mkdir()
    return root


class CombinedFrozenK8Controller(ScaleSearchController):
    """Run frozen I, C, and M selectors at their separate NSGA-II stages.

    C uses its frozen complete-pair representation for inference, but returns
    the selected pre-mutation crossover pair to the normal mutation stage.
    M therefore mutates each child at most once.
    """

    controls_initialization = True
    controls_crossover = True
    controls_mutation = True

    def __init__(self, *args, i_bundle, c_bundle, reliable_options, arcs,
                 **kwargs):
        batches = tuple(kwargs["batches"])
        normalizer = kwargs["archive"].normalizer
        super().__init__(*args, **kwargs)
        root = create_component_directories(self.out)
        i_policy = FrozenIResidualGate(read_json(i_bundle), digest)
        c_policy = FrozenCResidualGate(read_json(c_bundle), digest)
        common = {
            "method": self.method,
            "instance": self.instance,
            "seeds": self.seeds,
            "normalizer": normalizer,
        }
        self.i_controller = InitializationK8Controller(
            root / "I", self.run_id, self.scenario_id, self.budget,
            selection_mode="transformer", selection_policy=i_policy,
            reliable_options=reliable_options, batches=batches, arcs=arcs,
            pool_size=8, **common)
        self.c_controller = CrossoverK8Controller(
            root / "C", self.run_id, self.scenario_id, self.budget,
            selection_mode="transformer", selection_policy=c_policy,
            reliable_options=reliable_options, batches=batches, arcs=arcs,
            pool_size=8, defer_selected_pair_mutation=True, **common)
        self._sync_children()

    def _sync_children(self):
        for controller in (self.i_controller, self.c_controller):
            controller.generation = self.generation
            controller.phase = self.phase
            controller.evaluations = self.evaluations
            controller.cache_hits = self.cache_hits

    def initial_individual_from_base(self, *args, **kwargs):
        self._sync_children()
        return self.i_controller.initial_individual_from_base(*args, **kwargs)

    def initial_individual(self, *args, **kwargs):
        self._sync_children()
        return self.i_controller.initial_individual(*args, **kwargs)

    def evaluate(self, function, ind, *args, **kwargs):
        active = getattr(self.i_controller, "_active", None)
        if active is not None and id(ind) == active["placeholder_id"]:
            self._sync_children()
            previous_seconds = self.i_controller.objective_evaluation_seconds
            result = self.i_controller.evaluate(function, ind, *args, **kwargs)
            self.evaluations = self.i_controller.evaluations
            self.objective_evaluation_seconds += (
                self.i_controller.objective_evaluation_seconds
                - previous_seconds)
            self.archive.observe(
                ind.objectives, bool(ind.feasible), fingerprint(ind))
            if (self.generation < 0
                    and len(self.current_population) < self.pop_size):
                self.current_population.append(ind)
            self._sync_children()
            return result
        result = super().evaluate(function, ind, *args, **kwargs)
        self._sync_children()
        return result

    def settle_initialization(self, *_args, **_kwargs):
        self._sync_children()

    def crossover_common_node(self, *args, **kwargs):
        self._sync_children()
        self.c_controller.current_population = list(self.current_population)
        return self.c_controller.crossover_common_node(*args, **kwargs)

    def process_crossover_pair(self, *args, **kwargs):
        self._sync_children()
        self.c_controller.current_population = list(self.current_population)
        return self.c_controller.process_crossover_pair(*args, **kwargs)

    def settle_crossover(self, *args, **kwargs):
        self._sync_children()
        return self.c_controller.settle_crossover(*args, **kwargs)

    def flush_events(self, population):
        super().flush_events(population)
        self.i_controller.current_population = list(population)
        self.c_controller.flush_events(population)
        self._sync_children()

    def log_initial_population(self, population):
        super().log_initial_population(population)
        self.i_controller.current_population = list(population)
        self.c_controller.current_population = list(population)
        self._sync_children()

    def log_generation(self, population, boost):
        super().log_generation(population, boost)
        self.i_controller.current_population = list(population)
        self.c_controller.current_population = list(population)
        self._sync_children()

    def finish_episode(self):
        self.i_controller.finish_episode()
        self.c_controller.finish_episode()
        super().finish_episode()

    def close(self):
        try:
            self.i_controller.close()
            self.c_controller.close()
        finally:
            super().close()

    def summary(self):
        mutation = super().summary()
        initialization = self.i_controller.summary()
        crossover = self.c_controller.summary()
        return {
            **mutation,
            "combined_k8_protocol": "frozen-i-c-m-common-seed-v1",
            "component_summaries": {
                "I": initialization,
                "C": crossover,
                "M": mutation,
            },
            "combined_hidden_candidate_objective_evaluations": (
                initialization.get(
                    "initialization_k8_hidden_candidate_objective_evaluations", 0)
                + crossover.get(
                    "crossover_k8_hidden_candidate_objective_evaluations", 0)
                + mutation.get("k8_scale_search", {}).get(
                    "hidden_candidate_objective_evaluations", 0)),
        }


__all__ = ["CombinedFrozenK8Controller", "create_component_directories"]
