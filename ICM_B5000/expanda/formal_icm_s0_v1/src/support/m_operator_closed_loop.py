"""Runtime-only policy lifecycle and timing support."""
import time
from multi_instance_logging import fingerprint
from transformer_ppo.controller import PPOController

class FrozenPathPolicy:

    component = "M"
    training = False
    updates = 0
    ppo_update_seconds = 0.
    update_records = ()
    pending = None
    sequence_policy_actions = 0
    inference_action_mode = "argmax"
    policy_seed = None
    def begin_sequence(self):
        self.sequence_depth = 1
    def end_sequence(self, reward, done=False):
        self.sequence_depth = 0
    def finish(self):
        self.sequence_depth = 0


class TimedArchive:
    def __init__(self, archive, owner):
        self.wrapped, self.owner = archive, owner

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def observe(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.wrapped.observe(*args, **kwargs)
        finally:
            self.owner.hv_seconds += time.perf_counter() - started


class CompactProfile:
    """Keep only initial fingerprints and aggregate timing, no per-action dumps."""
    def append(self, name, row):
        return None

    def log_initial_population(self, population):
        self.initial_fingerprints = [fingerprint(p) for p in population]
        super().log_initial_population(population)

    def log_search_progress(self, population, checkpoint):
        started = time.perf_counter()
        try:
            return super().log_search_progress(population, checkpoint)
        finally:
            self.hv_seconds += time.perf_counter() - started

    def summary(self):
        return {**super().summary(), "initial_population_fingerprints": self.initial_fingerprints,
                "profile_hv_seconds": self.hv_seconds, "profile_log_mode": "aggregate-only"}


class FrozenOperatorController(CompactProfile, PPOController):
    def __init__(self, *args, **kwargs):
        self.hv_seconds, self.initial_fingerprints = 0., []
        super().__init__(*args, **kwargs)
        if self.candidate_audit_states:
            raise ValueError("counterfactual auditing is disabled in closed-loop inference")
        self.archive = TimedArchive(self.archive, self)
        self.decision_counts["base_mixture_choice"] = 0

    def _next_control_branch(self):
        self.mixture_event_count += 1
        return self.generation >= 10, self.mixture_event_count

    def _operator_candidates(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return super()._operator_candidates(*args, **kwargs)
        finally:
            self.trainer.feature_construction_seconds += time.perf_counter() - started

    def _population_hypervolume(self, population):
        started = time.perf_counter()
        try:
            return super()._population_hypervolume(population)
        finally:
            self.hv_seconds += time.perf_counter() - started

    def summary(self):
        return {**super().summary(), "minimum_control_generation": 10,
                "closed_loop_model_evaluations": self.trainer.model_evaluations,
                "closed_loop_switches": self.trainer.switches,
                "closed_loop_action_counts": dict(self.trainer.action_counts),
                "closed_loop_unsupported_states": self.trainer.unsupported,
                "execution_contract": "operator_shared_v1"}
