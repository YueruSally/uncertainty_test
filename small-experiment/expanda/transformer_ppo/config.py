"""Frozen configuration schemas and seed namespaces."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Tuple


SCENARIOS: Dict[str, Tuple[int, float]] = {
    "S0": (20, 1.0),
    "S1": (10, 0.5),
    "S2": (10, 1.0),
    "S3": (10, 2.0),
    "S4": (20, 0.5),
    "S5": (20, 2.0),
    "S6": (30, 0.5),
    "S7": (30, 1.0),
    "S8": (30, 2.0),
    "S9": (50, 0.5),
    "S10": (50, 1.0),
    "S11": (50, 2.0),
}
TRAIN_CONFIGS = tuple(f"S{i}" for i in range(9))
TEST_CONFIGS = tuple(f"S{i}" for i in range(12))
OPERATORS = ("add", "del", "mod", "mode", "replace")
OPERATOR_TO_ID = {name: index for index, name in enumerate(OPERATORS)}


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    gae_horizon: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.12
    update_epochs: int = 3
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-5
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    hidden_dim: int = 128
    attention_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    actor_policy: str = "direct"
    residual_logit_limit: float = 2.0
    candidate_representation: str = "legacy"
    initialization_control: str = "auto"
    credit_assignment: str = "continuous"
    mutation_control: str = "location"

    def validate(self) -> None:
        if self.rollout_steps < self.gae_horizon:
            raise ValueError("rollout_steps must be at least gae_horizon")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("actor and critic learning rates must be positive")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy coefficient must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.actor_policy not in ("direct", "base_residual"):
            raise ValueError(
                "actor_policy must be 'direct' or 'base_residual'")
        if self.residual_logit_limit <= 0.0:
            raise ValueError("residual_logit_limit must be positive")
        if self.candidate_representation not in (
                "legacy", "path_v2", "path_v3", "path_v4"):
            raise ValueError(
                "candidate_representation must be legacy, path_v2, path_v3, or path_v4")
        if self.initialization_control not in ("auto", "path", "path_count"):
            raise ValueError(
                "initialization_control must be 'auto', 'path', or 'path_count'")
        if (self.initialization_control == "path_count"
                and self.candidate_representation != "path_v3"):
            raise ValueError(
                "path_count initialization control requires path_v3")
        if self.credit_assignment not in ("continuous", "operation_terminal"):
            raise ValueError(
                "credit_assignment must be 'continuous' or 'operation_terminal'")
        if self.mutation_control not in (
                "location", "operator_location", "replace_path"):
            raise ValueError(
                "mutation_control must be location, operator_location, or replace_path")
        if (self.mutation_control == "operator_location"
                and (self.candidate_representation != "path_v3"
                     or self.credit_assignment != "operation_terminal")):
            raise ValueError(
                "operator_location requires path_v3 and operation_terminal credit")
        if (self.mutation_control == "replace_path"
                and (self.candidate_representation != "path_v4"
                     or self.credit_assignment != "operation_terminal")):
            raise ValueError(
                "replace_path requires path_v4 and operation_terminal credit")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")

    def digest(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def episode_seeds(instance_index: int, policy_seed: int) -> dict:
    """Disjoint, reproducible namespaces for one training episode."""
    if instance_index < 0 or policy_seed < 0:
        raise ValueError("seeds and indices must be non-negative")
    base = 20_000_000 + policy_seed * 1_000_000 + instance_index * 10
    return {
        "algorithm": base + 1,
        "policy": base + 2,
        "ccp100": base + 3,
    }
