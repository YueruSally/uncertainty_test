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

    def validate(self) -> None:
        if self.rollout_steps < self.gae_horizon:
            raise ValueError("rollout_steps must be at least gae_horizon")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")
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
