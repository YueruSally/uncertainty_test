from collections import Counter
import hashlib
import random
from m_operator_closed_loop import FrozenPathPolicy, FrozenOperatorController
from multi_instance_policy import Decision
from transformer_ppo.config import PPOConfig

class FixedMixPolicy(FrozenPathPolicy):
    """Reuse only the frozen-policy protocol; never load weights or observations."""

    def __init__(self, instance, seed, mode_probability):
        if not 0 <= mode_probability <= 1:
            raise ValueError("mode probability must be within [0,1]")
        self.name = "fixed-balanced-mode-mixture"
        self.config = PPOConfig(candidate_representation="path_v3", mutation_control="operator",
                                credit_assignment="generation_terminal", dropout=0.)
        self.config.validate()
        self.mode_probability = float(mode_probability)
        namespace = f"fixed-balanced-mode-v1:{instance}:{seed}"
        self.selection_seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
        self.selection_rng = random.Random(self.selection_seed)
        self.checkpoint_provenance = {"kind": "fixed-mixture-no-checkpoint", "learning_enabled": False,
            "mode_probability": self.mode_probability, "selection_seed": self.selection_seed}
        self.sequence_depth = 0
        self.feature_construction_seconds = 0.
        self.observation_provider = None
        self.model_evaluations = self.switches = self.unsupported = 0
        self.requested_mode = self.mode_ineligible_fallbacks = self.decisions = 0
        self.action_counts = Counter()

    def choose(self, candidates, context=None, **kwargs):
        if len(candidates) != 5:
            raise ValueError("requires unchanged five-operator action order")
        legal = [bool(row["eligible"]) for row in candidates]
        # Separate from global search RNG and the executor's target RNG.
        draw = self.selection_rng.random()
        self.decisions += 1
        probabilities = [0.] * 5
        if not legal[4]:
            self.unsupported += 1
            action = None  # Same unsupported-state treatment in both methods.
        else:
            request_mode = draw < self.mode_probability
            self.requested_mode += int(request_mode)
            self.mode_ineligible_fallbacks += int(request_mode and not legal[3])
            action = 3 if request_mode and legal[3] else 4
            probabilities[3] = self.mode_probability if legal[3] else 0.
            probabilities[4] = 1. - probabilities[3]
            self.action_counts[str(action)] += 1
            self.switches += int(action == 3)
        return Decision(action, probabilities, [None] * 5,
            {"decision_kind": "fixed_mixture", "inference_seconds": 0., "switched": action == 3})


class FixedMixController(FrozenOperatorController):
    def __init__(self, *args, **kwargs):
        self.successful_actions = Counter()
        self.effective_actions = Counter()
        super().__init__(*args, **kwargs)

    def append(self, name, row):
        if name == "mutation_outcomes" and row.get("control_branch") == "operator_scheduler":
            if row.get("mutation_success"):
                self.successful_actions[row["operator"]] += 1
                self.effective_actions[row["operator"]] += int(row.get("decision_changed", False))
        # Aggregate statistics only; no repeated candidate/observation dumps.

    def summary(self):
        p = self.trainer
        selected = sum(p.action_counts.values())
        return {**super().summary(), "fixed_mix": {
            "mode_probability": p.mode_probability, "selection_seed": p.selection_seed,
            "post_g10_decisions": p.decisions, "supported_decisions": selected,
            "mode_requests": p.requested_mode, "mode_selected": p.switches,
            "actual_mode_fraction": p.switches / selected if selected else 0.,
            "mode_ineligible_fallbacks": p.mode_ineligible_fallbacks,
            "balanced_unavailable": p.unsupported,
            "successful_actions": dict(self.successful_actions), "effective_actions": dict(self.effective_actions)}}
