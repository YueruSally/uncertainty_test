"""On-policy clipped PPO for variable-length mutation candidate sets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.distributions import Categorical

from multi_instance_policy import Decision

from .models import build_model
from .observations import (
    CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES, pad_observations,
)


@dataclass
class Transition:
    observation: dict
    action: int
    old_log_probability: float
    value: float
    reward: float
    done: bool


class PPOTrainer:
    def __init__(self, architecture, config, observation_provider, device=None,
                 policy_seed=0, training=True, component="M", model_spec=None):
        config.validate()
        self.architecture, self.config = architecture, config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # The policy owns its random streams.  In particular, PPO minibatch and
        # action sampling must not consume NumPy/Python randomness used by NSGA-II.
        torch.manual_seed(int(policy_seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(policy_seed))
        self.action_generator = torch.Generator(device=self.device.type)
        self.action_generator.manual_seed(int(policy_seed) + 1)
        self.minibatch_rng = np.random.default_rng(int(policy_seed) + 2)
        self.component = str(component).upper()
        default_spec = {
            "parent_dim": len(PARENT_FEATURES),
            "candidate_dim": len(CANDIDATE_FEATURES),
            "global_dim": len(GLOBAL_FEATURES),
            "action_count": 5,
        }
        self.model_spec = dict(default_spec if model_spec is None else model_spec)
        required = {"parent_dim", "candidate_dim", "global_dim", "action_count"}
        graph = {"graph_node_dim", "graph_edge_dim"}
        if (set(self.model_spec) not in (required, required | graph)
                or any(int(self.model_spec[name]) < 1 for name in self.model_spec)):
            raise ValueError(
                "model_spec must contain four positive dimensions and, optionally, "
                "both positive graph dimensions")
        self.model_spec = {name: int(value) for name, value in self.model_spec.items()}
        self.model = build_model(architecture, config, self.model_spec).to(self.device)
        critic_ids = {id(p) for p in self.model.heads.critic_head.parameters()}
        actor = [p for p in self.model.parameters() if id(p) not in critic_ids]
        critic = list(self.model.heads.critic_head.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": actor, "lr": config.actor_learning_rate},
            {"params": critic, "lr": config.critic_learning_rate},
        ])
        self.observation_provider = observation_provider
        self.buffer: list[Transition] = []
        self.pending = None
        self.training = bool(training)
        self.updates = 0
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.checkpoint_provenance = None
        self.feature_construction_seconds = 0.0
        self.ppo_update_seconds = 0.0
        self.update_records = []

    @property
    def name(self):
        suffix = "" if self.component == "M" else f"-{self.component.lower()}"
        return f"{self.architecture}-ppo{suffix}-masked"

    def _distribution(self, observation):
        batch = pad_observations([observation], self.device)
        logits, value = self.model(batch)
        return Categorical(logits=logits), value

    def choose(self, candidates, context=None):
        if self.pending is not None:
            raise RuntimeError("previous policy action has not received an outcome")
        eligible = [i for i, row in enumerate(candidates) if row.get("eligible")]
        if not eligible:
            return Decision(None, [0.0] * len(candidates), [None] * len(candidates),
                            {"no_eligible_candidate": True, "fallback": None,
                             "decision_kind": "no_candidate"})
        if len(eligible) == 1:
            probabilities = [0.0] * len(candidates)
            probabilities[eligible[0]] = 1.0
            # A forced move is execution logic, not a sampled policy action.
            # Do not build an observation or create a pending PPO transition.
            return Decision(
                eligible[0], probabilities, [None] * len(candidates),
                {"no_eligible_candidate": False, "fallback": None,
                 "decision_kind": "single_candidate"})
        feature_started = time.perf_counter()
        observation = self.observation_provider(candidates, context or {})
        self.feature_construction_seconds += time.perf_counter() - feature_started
        # Never update in the middle of one environment action.  The first
        # learnable sub-decision of the next sequence is a safe rollout
        # boundary and supplies the bootstrap state for the previous rollout.
        safe_update_boundary = (
            self.sequence_depth == 0 or self.sequence_policy_actions == 0)
        if (self.training and safe_update_boundary
                and len(self.buffer) >= self.config.rollout_steps):
            self.update(self._value(observation))
        self.model.train(self.training)
        inference_started = time.perf_counter()
        with torch.no_grad():
            distribution, value = self._distribution(observation)
            action = (torch.multinomial(
                distribution.probs, 1, generator=self.action_generator).squeeze(-1)
                if self.training else torch.argmax(distribution.logits, dim=-1))
            log_probability = distribution.log_prob(action)
            probabilities = distribution.probs.squeeze(0).cpu().tolist()
            entropy = distribution.entropy().item()
        self.pending = {
            "observation": observation, "action": int(action.item()),
            "old_log_probability": float(log_probability.item()), "value": float(value.item())}
        if self.sequence_depth:
            self.sequence_policy_actions += 1
        return Decision(int(action.item()), probabilities, [None] * len(probabilities), {
            "no_eligible_candidate": False, "fallback": None,
            "decision_kind": "model_choice",
            "old_log_probability": float(log_probability.item()),
            "state_value": float(value.item()), "entropy": float(entropy),
            "policy_update": self.updates, "architecture": self.architecture,
            "inference_seconds": time.perf_counter() - inference_started,
        })

    def observe_outcome(self, reward, done=False):
        if self.pending is None:
            return
        if self.training:
            self.buffer.append(Transition(reward=float(reward), done=bool(done), **self.pending))
        self.pending = None

    def begin_sequence(self):
        """Defer PPO updates while one environment action has sub-decisions."""
        if self.pending is not None or self.sequence_depth:
            raise RuntimeError("cannot begin a nested or pending policy sequence")
        self.sequence_depth = 1
        self.sequence_policy_actions = 0

    def end_sequence(self, terminal_reward, done=False):
        if self.sequence_depth != 1:
            raise RuntimeError("policy decision sequences cannot be nested")
        self.observe_outcome(terminal_reward, done=done)
        self.sequence_depth = 0
        self.sequence_policy_actions = 0

    def _value(self, observation):
        with torch.no_grad():
            _, value = self._distribution(observation)
        return float(value.item())

    def finish(self):
        if self.sequence_depth or self.sequence_policy_actions:
            raise RuntimeError("episode ended inside a policy decision sequence")
        if self.pending is not None:
            raise RuntimeError("episode ended before pending action received a reward")
        if self.training and self.buffer:
            self.buffer[-1].done = True
            self.update(0.0)

    def _advantages(self, bootstrap_value):
        rewards = np.asarray([x.reward for x in self.buffer], dtype=np.float32)
        values = np.asarray([x.value for x in self.buffer] + [bootstrap_value], dtype=np.float32)
        dones = np.asarray([x.done for x in self.buffer], dtype=bool)
        deltas = rewards + self.config.gamma * values[1:] * (~dones) - values[:-1]
        advantages = np.zeros_like(rewards)
        for start in range(len(rewards)):
            factor = 1.0
            for offset in range(self.config.gae_horizon):
                index = start + offset
                if index >= len(rewards):
                    break
                advantages[start] += factor * deltas[index]
                if dones[index]:
                    break
                factor *= self.config.gamma * self.config.gae_lambda
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, bootstrap_value=0.0):
        if not self.training or not self.buffer:
            return {}
        update_started = time.perf_counter()
        transition_count = len(self.buffer)
        positive_reward_count = sum(item.reward > 0.0 for item in self.buffer)
        reward_total = float(sum(item.reward for item in self.buffer))
        advantages, returns = self._advantages(float(bootstrap_value))
        old_logp = torch.as_tensor([x.old_log_probability for x in self.buffer],
                                   dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in self.buffer],
                                  dtype=torch.long, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(self.buffer))
        diagnostics = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}
        stop = False
        for _ in range(self.config.update_epochs):
            indices = self.minibatch_rng.permutation(indices)
            for start in range(0, len(indices), 64):
                selected = indices[start:start + 64]
                observations = [self.buffer[i].observation for i in selected]
                batch = pad_observations(observations, self.device)
                logits, value = self.model(batch)
                distribution = Categorical(logits=logits)
                index = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                new_logp = distribution.log_prob(actions[index])
                ratio = torch.exp(new_logp - old_logp[index])
                unclipped = ratio * advantages_t[index]
                clipped = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                      1 + self.config.clip_ratio) * advantages_t[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - returns_t[index]).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = (policy_loss + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                approximate_kl = (old_logp[index] - new_logp).mean().item()
                diagnostics["policy_loss"].append(float(policy_loss.item()))
                diagnostics["value_loss"].append(float(value_loss.item()))
                diagnostics["entropy"].append(float(entropy.item()))
                diagnostics["kl"].append(float(approximate_kl))
                if approximate_kl > self.config.target_kl:
                    stop = True
                    break
            if stop:
                break
        self.buffer.clear()
        self.updates += 1
        means = {
            name: float(np.mean(values))
            for name, values in diagnostics.items() if values
        }
        elapsed = time.perf_counter() - update_started
        self.ppo_update_seconds += elapsed
        self.update_records.append({
            "update": self.updates,
            "transition_count": transition_count,
            "positive_reward_transition_count": positive_reward_count,
            "reward_total": reward_total,
            "update_seconds": elapsed,
            **means,
        })
        return means

    def save(self, path, metadata=None, overwrite=False):
        """Atomically save a checkpoint at an episode boundary.

        A training checkpoint is resumable only after ``finish`` has consumed
        the rollout and cleared the pending action.  Refusing an in-flight
        checkpoint avoids silently dropping PPO experience after a restart.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise ValueError(f"checkpoint already exists: {path}")
        if (self.pending is not None or self.buffer or self.sequence_depth
                or self.sequence_policy_actions):
            raise RuntimeError("checkpoint requires an empty PPO rollout boundary")
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 3,
            "architecture": self.architecture,
            "component": self.component,
            "model_spec": self.model_spec,
            "ppo_config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "metadata": metadata or {},
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda_all": (torch.cuda.get_rng_state_all()
                             if self.device.type == "cuda" else None),
                "action_generator": self.action_generator.get_state(),
                "minibatch_numpy": self.minibatch_rng.bit_generator.state,
            },
        }
        temporary = path.with_name(path.name + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            torch.save(artifact, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path, config_type, observation_provider, device=None, training=False):
        path = Path(path)
        try:
            artifact = torch.load(
                path, map_location=device or "cpu", weights_only=False)
        except TypeError:  # PyTorch before the weights_only argument was added.
            artifact = torch.load(path, map_location=device or "cpu")
        if artifact.get("schema_version") not in (1, 2, 3):
            raise ValueError("unsupported checkpoint schema")
        config = config_type(**artifact["ppo_config"])
        result = cls(
            artifact["architecture"], config, observation_provider,
            device=device, training=training,
            component=artifact.get("component", "M"),
            model_spec=artifact.get("model_spec"))
        result.model.load_state_dict(artifact["model"])
        if training:
            result.optimizer.load_state_dict(artifact["optimizer"])
            if artifact.get("schema_version") < 2:
                raise ValueError("schema-1 checkpoints cannot resume training reproducibly")
            rng = artifact.get("rng", {})
            required = {"torch", "cuda_all", "action_generator", "minibatch_numpy"}
            if set(rng) != required:
                raise ValueError("resumable checkpoint has incomplete random-number state")
            torch.set_rng_state(rng["torch"].cpu())
            if result.device.type == "cuda":
                if rng["cuda_all"] is None:
                    raise ValueError("CUDA resume requested from a CPU checkpoint")
                if len(rng["cuda_all"]) != torch.cuda.device_count():
                    raise ValueError("visible CUDA device count changed since checkpoint")
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda_all"]])
            result.action_generator.set_state(rng["action_generator"].cpu())
            result.minibatch_rng.bit_generator.state = rng["minibatch_numpy"]
        result.updates = int(artifact.get("updates", 0))
        metadata = artifact.get("metadata") or {}
        metadata_blob = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result.checkpoint_provenance = {
            "checkpoint_schema_version": int(artifact["schema_version"]),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "training_metadata_sha256": hashlib.sha256(
                metadata_blob.encode("utf-8")).hexdigest(),
            "architecture": artifact["architecture"],
            "component": result.component,
            "model_spec": result.model_spec,
            "policy_seed": metadata.get("policy_seed"),
            "catalog_digest": metadata.get("catalog_digest"),
            "normalization_digest": metadata.get("normalization_digest"),
            "training_episode_count": metadata.get("episode_count"),
            "ppo_digest": metadata.get("ppo_digest"),
            "updates": result.updates,
        }
        return result
