#!/usr/bin/env python3
"""Run Base, one frozen component, or a frozen component combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig
from transformer_ppo.episode import run_episode
from transformer_ppo.ppo import PPOTrainer


def _normalizer(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(
        tuple(value["minimum"]), tuple(value["maximum"]),
        tuple(value.get("reference", (1.1, 1.1, 1.1))))


def _load(path, component, device, inference_action_mode,
          inference_policy_seed):
    if path is None or not path.is_file():
        raise ValueError(f"{component} checkpoint is missing: {path}")
    trainer = PPOTrainer.load(
        path, PPOConfig, lambda *_: None, device=device, training=False,
        inference_action_mode=inference_action_mode,
        inference_policy_seed=inference_policy_seed)
    if trainer.component != component:
        raise ValueError(
            f"checkpoint {path} is component {trainer.component}, expected {component}")
    return trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--component", choices=(
        "BASE", "RULE-I", "I", "C", "M", "IC", "IM", "CM", "ICM"),
                        required=True)
    parser.add_argument("--initialization-rule-normalization", type=Path)
    parser.add_argument("--initialization-rule-top-k", type=int,
                        help="V2 learned-I Rule-I prefilter size.")
    parser.add_argument("--model-choice-probability", type=float, default=1.0,
                        help="V2 single C/M probability of using the model.")
    parser.add_argument("--c-model-choice-probability", type=float, default=1.0)
    parser.add_argument("--m-model-choice-probability", type=float, default=1.0)
    parser.add_argument(
        "--model-choice-event", type=int,
        help=("M-only diagnostic: use the model exactly at this global "
              "mutation-call index and Base at every other call."))
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint for a single I, C, or M run.")
    parser.add_argument("--i-checkpoint", type=Path)
    parser.add_argument("--c-checkpoint", type=Path)
    parser.add_argument("--m-checkpoint", type=Path)
    parser.add_argument("--algorithm-seed", type=int, required=True)
    parser.add_argument(
        "--inference-action-mode", choices=("argmax", "sample"),
        default="argmax",
        help="Frozen-policy action selection; does not enable PPO training.")
    parser.add_argument(
        "--inference-policy-seed", type=int, default=0,
        help="Private action-sampling seed used only when mode=sample.")
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument(
        "--candidate-audit-states", type=int, default=0,
        help=("M-only fixed-state diagnostic states; extra counterfactual "
              "evaluations do not consume the NSGA-II budget."))
    parser.add_argument("--candidate-audit-candidates", type=int, default=4)
    parser.add_argument("--candidate-audit-replicates", type=int, default=1)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    try:
        normalizer = _normalizer(args.normalization)
        rule_normalization = None
        needs_rule_normalization = (
            args.component == "RULE-I"
            or args.initialization_rule_top_k is not None)
        if needs_rule_normalization:
            if (args.initialization_rule_normalization is None
                    or not args.initialization_rule_normalization.is_file()):
                raise ValueError(
                    "RULE-I/I prefilter requires "
                    "--initialization-rule-normalization")
            rule_normalization = json.loads(
                args.initialization_rule_normalization.read_text(encoding="utf-8"))
        if args.initialization_rule_top_k is not None:
            if "I" not in args.component:
                raise ValueError("I rule top-k requires an active I component")
            if args.initialization_rule_top_k < 2:
                raise ValueError("I rule top-k must be at least 2")
        for name, value in (
                ("model", args.model_choice_probability),
                ("C", args.c_model_choice_probability),
                ("M", args.m_model_choice_probability)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} choice probability must be in [0, 1]")
        if args.candidate_audit_states < 0:
            raise ValueError("candidate audit states cannot be negative")
        if args.candidate_audit_candidates < 2:
            raise ValueError("candidate audit candidates must be at least 2")
        if args.candidate_audit_replicates < 1:
            raise ValueError("candidate audit replicates must be positive")
        if args.candidate_audit_states and args.component != "M":
            raise ValueError("fixed-state candidate audit currently supports only M")
        if args.model_choice_event is not None:
            if args.model_choice_event < 1:
                raise ValueError("model choice event must be a positive integer")
            if args.component != "M":
                raise ValueError("model choice event currently supports only M")
            if args.model_choice_probability != 1.0:
                raise ValueError(
                    "model choice event cannot be combined with probability gating")
        if args.component in ("BASE", "RULE-I"):
            trainer = None
        elif len(args.component) > 1:
            checkpoint_paths = {
                "I": args.i_checkpoint,
                "C": args.c_checkpoint,
                "M": args.m_checkpoint,
            }
            trainer = {
                name: _load(
                    checkpoint_paths[name], name, args.device,
                    args.inference_action_mode,
                    args.inference_policy_seed + "ICM".index(name) + 1)
                for name in args.component
            }
        else:
            trainer = _load(
                args.checkpoint, args.component, args.device,
                args.inference_action_mode, args.inference_policy_seed)
        summary = run_episode(
            args.data, args.instance, args.out, trainer, normalizer,
            algorithm_seed=args.algorithm_seed,
            evaluation_budget=args.evaluation_budget, pop_size=args.pop,
            generations=args.gens, reward_config=RewardConfig(),
            evaluation_mode="deterministic", component=args.component,
            rule_initialization_normalization=(
                rule_normalization if args.component == "RULE-I" else None),
            initialization_learning_rule_normalization=(
                rule_normalization
                if args.initialization_rule_top_k is not None else None),
            initialization_learning_rule_top_k=args.initialization_rule_top_k,
            model_choice_event=args.model_choice_event,
            model_choice_probabilities=(
                {args.component: args.model_choice_probability}
                if args.component in ("C", "M") else
                {
                    name: value for name, value in (
                        ("C", args.c_model_choice_probability),
                        ("M", args.m_model_choice_probability),
                    ) if name in args.component
                }
                if len(args.component) > 1 else None),
            candidate_audit=(
                {
                    "states": args.candidate_audit_states,
                    "candidates": args.candidate_audit_candidates,
                    "replicates": args.candidate_audit_replicates,
                }
                if args.candidate_audit_states else None))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
