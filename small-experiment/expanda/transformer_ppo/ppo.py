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

from .action_groups import action_group_diagnostics, grouped_argmax
from .models import TRANSFORMER_ACTOR_CONDITIONING, build_model
from .observations import (
    CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES, pad_observations,
)
from .supervised_prior import SupervisedMutationPrior


@dataclass
class Transition:
    observation: dict
    action: int
    old_log_probability: float
    old_probabilities: tuple[float, ...]
    value: float
    reward: float
    done: bool


class PPOTrainer:
    def __init__(self, architecture, config, observation_provider, device=None,
                 policy_seed=0, training=True, component="M", model_spec=None,
                 inference_action_mode="argmax", supervised_prior=None,
                 grouped_argmax_margin=0.0):
        config.validate()
        if inference_action_mode not in ("argmax", "sample"):
            raise ValueError(
                "inference_action_mode must be 'argmax' or 'sample'")
        if not 0.0 <= float(grouped_argmax_margin) < 1.0:
            raise ValueError("grouped_argmax_margin must be in [0, 1)")
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
        self.inference_action_mode = str(inference_action_mode)
        self.grouped_argmax_margin = float(grouped_argmax_margin)
        self.policy_seed = int(policy_seed)
        self.updates = 0
        self.sequence_depth = 0
        self.sequence_policy_actions = 0
        self.checkpoint_provenance = None
        self.feature_construction_seconds = 0.0
        self.ppo_update_seconds = 0.0
        self.update_records = []
        self.supervised_prior = (
            None if supervised_prior is None
            else SupervisedMutationPrior(supervised_prior))

    @property
    def name(self):
        suffix = "" if self.component == "M" else f"-{self.component.lower()}"
        return f"{self.architecture}-ppo{suffix}-masked"

    def _distribution(self, observation):
        batch = pad_observations([observation], self.device)
        logits, value = self.model(batch)
        return Categorical(logits=logits), value

    def choose(self, candidates, context=None, forced_action_index=None):
        if self.pending is not None:
            raise RuntimeError("previous policy action has not received an outcome")
        eligible = [i for i, row in enumerate(candidates) if row.get("eligible")]
        forced = (
            None if forced_action_index is None else int(forced_action_index))
        if forced is not None and self.training:
            raise ValueError(
                "forced action index is a frozen-policy diagnostic only")
        if forced is not None and (forced < 0 or forced >= len(candidates)):
            raise ValueError(
                f"forced action index {forced} is outside {len(candidates)} candidates")
        if forced is not None and forced not in eligible:
            raise ValueError(f"forced action index {forced} is not eligible")
        if not eligible:
            return Decision(None, [0.0] * len(candidates), [None] * len(candidates),
                            {"no_eligible_candidate": True, "fallback": None,
                             "decision_kind": "no_candidate"})
        if len(eligible) == 1 and forced is None:
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
        prior_diagnostics = None
        if self.supervised_prior is not None:
            observation, prior_diagnostics = self.supervised_prior.apply(observation)
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
            sample_action = (
                forced is None
                and (self.training or self.inference_action_mode == "sample"))
            probabilities_tensor = distribution.probs.squeeze(0)
            probabilities = probabilities_tensor.cpu().tolist()
            group_diagnostics = action_group_diagnostics(
                candidates, probabilities)
            if forced is not None:
                action = torch.as_tensor(
                    [forced], dtype=torch.long, device=self.device)
            elif sample_action:
                action = torch.multinomial(
                    distribution.probs, 1,
                    generator=self.action_generator).squeeze(-1)
            else:
                action_index, grouped_diagnostics = grouped_argmax(
                    candidates, probabilities, self.grouped_argmax_margin)
                action = torch.as_tensor(
                    [action_index], dtype=torch.long, device=self.device)
                if grouped_diagnostics is not None:
                    group_diagnostics = grouped_diagnostics
            log_probability = distribution.log_prob(action)
            entropy = distribution.entropy().item()
        self.pending = {
            "observation": observation, "action": int(action.item()),
            "old_log_probability": float(log_probability.item()), "value": float(value.item())}
        self.pending["old_probabilities"] = tuple(float(x) for x in probabilities)
        if self.sequence_depth:
            self.sequence_policy_actions += 1
        selected_group = (
            None if group_diagnostics is None else
            group_diagnostics["groups"][int(action.item())])
        return Decision(int(action.item()), probabilities, [None] * len(probabilities), {
            "no_eligible_candidate": False, "fallback": None,
            "decision_kind": "model_choice",
            "old_log_probability": float(log_probability.item()),
            "state_value": float(value.item()), "entropy": float(entropy),
            "actor_policy": self.config.actor_policy,
            "candidate_representation": self.config.candidate_representation,
            "initialization_control": self.config.initialization_control,
            "credit_assignment": self.config.credit_assignment,
            "mutation_control": self.config.mutation_control,
            "supervised_prior": prior_diagnostics,
            "policy_update": self.updates, "architecture": self.architecture,
            "action_group_probabilities": (
                None if group_diagnostics is None else
                group_diagnostics["masses"]),
            "selected_action_group": selected_group,
            "grouped_argmax": bool(
                not sample_action and group_diagnostics is not None),
            "grouped_argmax_margin": self.grouped_argmax_margin,
            "forced_action_index": forced,
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
        if self.pending is not None:
            self.observe_outcome(terminal_reward, done=done)
        elif self.training and self.sequence_policy_actions:
            # A final forced sub-action creates no PPO transition.  In that
            # case the most recent learnable action is the terminal action and
            # must receive the operation reward instead of retaining its
            # temporary zero credit.
            self.buffer[-1].reward = float(terminal_reward)
            self.buffer[-1].done = bool(done)
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
        rewards = np.asarray([item.reward for item in self.buffer], dtype=np.float64)
        positive_reward_count = int(np.count_nonzero(rewards > 0.0))
        nonzero_reward_count = int(np.count_nonzero(rewards != 0.0))
        reward_total = float(rewards.sum())
        advantages, returns = self._advantages(float(bootstrap_value))
        old_logp = torch.as_tensor([x.old_log_probability for x in self.buffer],
                                   dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([x.action for x in self.buffer],
                                  dtype=torch.long, device=self.device)
        advantages_t = torch.as_tensor(advantages, device=self.device)
        returns_t = torch.as_tensor(returns, device=self.device)
        indices = np.arange(len(self.buffer))
        diagnostics = {
            "policy_loss": [], "value_loss": [], "entropy": [], "kl": [],
            "absolute_log_probability_change": [],
            "clip_fraction": [], "actor_head_gradient_norm": [],
            "critic_head_gradient_norm": [], "shared_gradient_norm": [],
        }
        actor_head_ids = {
            id(parameter)
            for parameter in self.model.heads.actor_heads.parameters()
        }
        critic_head_ids = {
            id(parameter)
            for parameter in self.model.heads.critic_head.parameters()
        }
        actor_head_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) in actor_head_ids
        ]
        critic_head_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) in critic_head_ids
        ]
        shared_parameters = [
            parameter for parameter in self.model.parameters()
            if id(parameter) not in actor_head_ids | critic_head_ids
        ]

        def gradient_norm(parameters):
            squared = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in parameters if parameter.grad is not None)
            return squared ** 0.5
        first_minibatch_pre_step = None
        minibatch_count = 0
        update_epochs_started = 0
        stop = False
        for _ in range(self.config.update_epochs):
            update_epochs_started += 1
            indices = self.minibatch_rng.permutation(indices)
            for start in range(0, len(indices), 64):
                selected = indices[start:start + 64]
                observations = [self.buffer[i].observation for i in selected]
                batch = pad_observations(observations, self.device)
                logits, value = self.model(batch)
                distribution = Categorical(logits=logits)
                index = torch.as_tensor(selected, dtype=torch.long, device=self.device)
                new_logp = distribution.log_prob(actions[index])
                logp_difference = new_logp - old_logp[index]
                ratio = torch.exp(logp_difference)
                if first_minibatch_pre_step is None:
                    absolute_error = logp_difference.detach().abs()
                    ratio_error = (ratio.detach() - 1.0).abs()
                    first_minibatch_pre_step = {
                        "first_minibatch_size": int(len(selected)),
                        "first_minibatch_pre_step_logp_abs_error_mean": float(
                            absolute_error.mean().item()),
                        "first_minibatch_pre_step_logp_abs_error_max": float(
                            absolute_error.max().item()),
                        "first_minibatch_pre_step_ratio_abs_error_mean": float(
                            ratio_error.mean().item()),
                        "first_minibatch_pre_step_ratio_abs_error_max": float(
                            ratio_error.max().item()),
                    }
                unclipped = ratio * advantages_t[index]
                clipped = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                      1 + self.config.clip_ratio) * advantages_t[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - returns_t[index]).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = (policy_loss + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy)
                approximate_kl = (old_logp[index] - new_logp).mean().item()
                self.optimizer.zero_grad()
                loss.backward()
                diagnostics["actor_head_gradient_norm"].append(
                    gradient_norm(actor_head_parameters))
                diagnostics["critic_head_gradient_norm"].append(
                    gradient_norm(critic_head_parameters))
                diagnostics["shared_gradient_norm"].append(
                    gradient_norm(shared_parameters))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                minibatch_count += 1
                diagnostics["policy_loss"].append(float(policy_loss.item()))
                diagnostics["value_loss"].append(float(value_loss.item()))
                diagnostics["entropy"].append(float(entropy.item()))
                diagnostics["kl"].append(float(approximate_kl))
                diagnostics["absolute_log_probability_change"].append(float(
                    logp_difference.detach().abs().mean().item()))
                diagnostics["clip_fraction"].append(float(
                    ((ratio.detach() - 1.0).abs() > self.config.clip_ratio)
                    .to(torch.float32).mean().item()))
                if approximate_kl > self.config.target_kl:
                    stop = True
                    break
            if stop:
                break
        post_update_kl = []
        post_update_total_variation = []
        post_update_top_action_changed = []
        post_update_values = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(self.buffer), 64):
                rows = self.buffer[start:start + 64]
                batch = pad_observations(
                    [item.observation for item in rows], self.device)
                logits, values = self.model(batch)
                probabilities = torch.softmax(logits, dim=-1).cpu()
                post_update_values.extend(values.detach().cpu().tolist())
                for row_index, item in enumerate(rows):
                    old = torch.as_tensor(
                        item.old_probabilities, dtype=torch.float64)
                    new = probabilities[
                        row_index, :len(item.old_probabilities)].to(torch.float64)
                    support = old > 0.0
                    post_update_kl.append(float((
                        old[support]
                        * (old[support].log()
                           - new[support].clamp_min(1e-15).log())
                    ).sum().item()))
                    post_update_total_variation.append(float(
                        0.5 * (old - new).abs().sum().item()))
                    post_update_top_action_changed.append(float(
                        int(old.argmax().item() != new.argmax().item())))
        self.model.train(was_training)
        return_variance = float(np.var(returns))
        explained_variance = (
            None if return_variance <= 1e-12 else
            float(1.0 - np.var(
                returns - np.asarray(post_update_values, dtype=np.float32))
                / return_variance))
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
            "nonzero_reward_transition_count": nonzero_reward_count,
            "zero_reward_transition_count": transition_count - nonzero_reward_count,
            "reward_total": reward_total,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "return_variance": return_variance,
            "critic_explained_variance": explained_variance,
            "post_update_full_distribution_kl_mean": float(
                np.mean(post_update_kl)),
            "post_update_full_distribution_kl_max": float(
                np.max(post_update_kl)),
            "post_update_total_variation_mean": float(
                np.mean(post_update_total_variation)),
            "post_update_total_variation_max": float(
                np.max(post_update_total_variation)),
            "post_update_top_action_change_fraction": float(
                np.mean(post_update_top_action_changed)),
            "minibatch_count": minibatch_count,
            "optimizer_step_count": minibatch_count,
            "configured_update_epochs": int(self.config.update_epochs),
            "update_epochs_started": update_epochs_started,
            "early_stopped_on_target_kl": stop,
            "update_seconds": elapsed,
            **(first_minibatch_pre_step or {}),
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
            "schema_version": 5,
            "architecture": self.architecture,
            "actor_conditioning": (
                TRANSFORMER_ACTOR_CONDITIONING
                if self.architecture == "transformer" else None),
            "component": self.component,
            "model_spec": self.model_spec,
            "ppo_config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "supervised_prior": (
                None if self.supervised_prior is None
                else self.supervised_prior.artifact),
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
    def load(cls, path, config_type, observation_provider, device=None,
             training=False, inference_action_mode="argmax",
             inference_policy_seed=0, grouped_argmax_margin=0.0):
        path = Path(path)
        try:
            artifact = torch.load(
                path, map_location=device or "cpu", weights_only=False)
        except TypeError:  # PyTorch before the weights_only argument was added.
            artifact = torch.load(path, map_location=device or "cpu")
        if artifact.get("schema_version") not in (1, 2, 3, 4, 5):
            raise ValueError("unsupported checkpoint schema")
        if (artifact.get("architecture") == "transformer"
                and artifact.get("actor_conditioning")
                != TRANSFORMER_ACTOR_CONDITIONING):
            raise ValueError(
                "Transformer checkpoint predates global-conditioned actor; "
                "retrain it before evaluation or resume")
        config = config_type(**artifact["ppo_config"])
        result = cls(
            artifact["architecture"], config, observation_provider,
            device=device, policy_seed=inference_policy_seed,
            training=training, inference_action_mode=inference_action_mode,
            grouped_argmax_margin=grouped_argmax_margin,
            component=artifact.get("component", "M"),
            model_spec=artifact.get("model_spec"),
            supervised_prior=artifact.get("supervised_prior"))
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
            "actor_conditioning": artifact.get("actor_conditioning"),
            "component": result.component,
            "model_spec": result.model_spec,
            "policy_seed": metadata.get("policy_seed"),
            "catalog_digest": metadata.get("catalog_digest"),
            "normalization_digest": metadata.get("normalization_digest"),
            "training_episode_count": metadata.get("episode_count"),
            "ppo_digest": metadata.get("ppo_digest"),
            "actor_policy": config.actor_policy,
            "candidate_representation": config.candidate_representation,
            "initialization_control": config.initialization_control,
            "credit_assignment": config.credit_assignment,
            "mutation_control": config.mutation_control,
            "supervised_prior_digest": (
                None if result.supervised_prior is None
                else result.supervised_prior.digest),
            "learning_control": metadata.get("learning_control"),
            "updates": result.updates,
        }
        return result
