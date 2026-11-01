#!/usr/bin/env python3
"""Run Base, one frozen component, or a frozen component combination."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig
from transformer_ppo.episode import CCP_UNCERTAINTY_PROFILES, run_episode
from transformer_ppo.initialization_rule import INITIALIZATION_RULE_PROFILES
from transformer_ppo.ppo import PPOTrainer


def _normalizer(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(
        tuple(value["minimum"]), tuple(value["maximum"]),
        tuple(value.get("reference", (1.1, 1.1, 1.1))))


def _load(path, component, device, inference_action_mode,
          inference_policy_seed, grouped_argmax_margin):
    if path is None or not path.is_file():
        raise ValueError(f"{component} checkpoint is missing: {path}")
    trainer = PPOTrainer.load(
        path, PPOConfig, lambda *_: None, device=device, training=False,
        inference_action_mode=inference_action_mode,
        inference_policy_seed=inference_policy_seed,
        grouped_argmax_margin=grouped_argmax_margin)
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
    parser.add_argument(
        "--initialization-rule-profile",
        choices=INITIALIZATION_RULE_PROFILES, default="legacy",
        help=("RULE-I path selector. Directed profiles use within-batch "
              "cost/emission/time/lateness/reliability scaling."))
    parser.add_argument("--initialization-rule-top-k", type=int,
                        help="V2 learned-I Rule-I prefilter size.")
    parser.add_argument("--model-choice-probability", type=float, default=1.0,
                        help="V2 single C/M probability of using the model.")
    parser.add_argument("--c-model-choice-probability", type=float, default=1.0)
    parser.add_argument("--m-model-choice-probability", type=float, default=1.0)
    parser.add_argument(
        "--model-choice-event", type=int,
        help=("I/C/M one-shot diagnostic: use the model exactly at this global "
              "component-call index and Base at every other call."))
    parser.add_argument(
        "--i-forced-candidate-index", type=int,
        help=("I-only one-shot diagnostic: at --model-choice-event execute "
              "this deterministic path candidate index instead of inference."))
    parser.add_argument(
        "--c-forced-joint-candidate-index", type=int,
        help=("C-only one-shot diagnostic: at --model-choice-event execute "
              "this deterministic joint candidate index instead of sampling."))
    parser.add_argument(
        "--m-forced-candidate-index", type=int,
        help=("M-only one-shot diagnostic: at --model-choice-event execute "
              "this deterministic raw candidate index instead of inference."))
    parser.add_argument(
        "--m-forced-operator", choices=("add", "del", "mod", "mode", "replace"),
        help=("M-only one-shot operator diagnostic: at --model-choice-event "
              "force this operator while retaining Base batch and internal "
              "target selection."))
    parser.add_argument(
        "--m-forced-operator-min-generation", type=int,
        help=("M-only schedule: from this generation onward, force "
              "--m-forced-operator according to its schedule probability."))
    parser.add_argument(
        "--m-forced-operator-probability", type=float, default=1.0,
        help="Probability of forcing the scheduled M operator (default: 1).")
    parser.add_argument(
        "--m-forced-operator-positive-probe", action="store_true",
        help=("After the first scheduled forced operator, continue forcing "
              "only when that probe immediately increases q90 archive HV."))
    parser.add_argument(
        "--m-forced-operator-reversible-probe", action="store_true",
        help=("Count and preview the first forced-operator evaluation; when "
              "its archive-HV delta is nonpositive, restore and execute the "
              "exact Base mutation instead."))
    parser.add_argument(
        "--m-forced-operator-probe-max-attempts", type=int, default=1,
        help="Maximum number of reversible operator-probe attempts.")
    parser.add_argument(
        "--m-forced-operator-probe-retry-events", type=int, default=0,
        help=("Mutation events to skip before retrying a rejected reversible "
              "operator probe."))
    parser.add_argument(
        "--m-forced-operator-best-of", type=int, default=1,
        help=("For a scheduled forced Replace, generate and q90-CCP evaluate "
              "this many distinct budget-counted candidates, then execute "
              "the best (default: 1)."))
    parser.add_argument(
        "--m-forced-replace-selector",
        choices=("base_random", "joint_extreme_cycle", "joint_balanced",
                 "joint_pareto_cycle"),
        default="base_random",
        help=("Zero-extra-evaluation joint batch/path selector used by a "
              "scheduled forced Replace."))
    parser.add_argument(
        "--m-continuation-event", type=int,
        help=("M-only paired-continuation diagnostic: after this global "
              "mutation call, start --m-continuation-seed."))
    parser.add_argument(
        "--m-continuation-seed", type=int,
        help="M-only post-intervention Python/NumPy continuation seed.")
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
    parser.add_argument(
        "--inference-group-margin", type=float, default=0.0,
        help=("Grouped argmax safety margin. For policies declaring a safe "
              "action group, an alternative group must exceed its total "
              "probability by this margin or inference falls back to safe."))
    parser.add_argument("--evaluation-budget", type=int, default=15_000)
    parser.add_argument(
        "--evaluation-mode", choices=("deterministic", "ccp100", "ccp"),
        default="deterministic")
    parser.add_argument("--scenario-count", type=int)
    parser.add_argument("--ccp-confidence", type=float, default=0.90)
    parser.add_argument(
        "--uncertainty-profile",
        choices=tuple(CCP_UNCERTAINTY_PROFILES), default="legacy")
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
            (args.component == "RULE-I"
             and args.initialization_rule_profile == "legacy")
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
        if not 0.0 <= args.inference_group_margin < 1.0:
            raise ValueError("inference group margin must be in [0, 1)")
        if args.model_choice_event is not None:
            if args.model_choice_event < 1:
                raise ValueError("model choice event must be a positive integer")
            if args.component not in ("I", "C", "M"):
                raise ValueError("model choice event supports only I, C, or M")
            if args.model_choice_probability != 1.0:
                raise ValueError(
                    "model choice event cannot be combined with probability gating")
        if args.i_forced_candidate_index is not None:
            if args.i_forced_candidate_index < 0:
                raise ValueError("forced I candidate index cannot be negative")
            if args.component != "I" or args.model_choice_event is None:
                raise ValueError(
                    "forced I candidate index requires component I and "
                    "--model-choice-event")
        if args.c_forced_joint_candidate_index is not None:
            if args.c_forced_joint_candidate_index < 0:
                raise ValueError("forced C joint candidate index cannot be negative")
            if args.component != "C" or args.model_choice_event is None:
                raise ValueError(
                    "forced C joint candidate index requires component C and "
                    "--model-choice-event")
        if args.m_forced_candidate_index is not None:
            if args.m_forced_candidate_index < 0:
                raise ValueError("forced M candidate index cannot be negative")
            if args.component != "M" or args.model_choice_event is None:
                raise ValueError(
                    "forced M candidate index requires component M and "
                    "--model-choice-event")
        if args.m_forced_operator is not None:
            if args.component != "M":
                raise ValueError(
                    "forced M operator requires component M")
            if ((args.model_choice_event is None)
                    == (args.m_forced_operator_min_generation is None)):
                raise ValueError(
                    "forced M operator requires exactly one of "
                    "--model-choice-event and "
                    "--m-forced-operator-min-generation")
            if args.m_forced_candidate_index is not None:
                raise ValueError(
                    "forced M operator and candidate index are mutually exclusive")
            if not 0.0 <= args.m_forced_operator_probability <= 1.0:
                raise ValueError(
                    "forced M operator probability must be in [0, 1]")
            if (args.m_forced_operator_positive_probe
                    and args.m_forced_operator_probability != 1.0):
                raise ValueError(
                    "positive operator probe requires probability 1")
            if (args.m_forced_operator_reversible_probe
                    and not args.m_forced_operator_positive_probe):
                raise ValueError(
                    "reversible operator probe requires positive probe")
            if args.m_forced_operator_probe_max_attempts < 1:
                raise ValueError(
                    "operator probe maximum attempts must be positive")
            if args.m_forced_operator_probe_retry_events < 0:
                raise ValueError(
                    "operator probe retry events cannot be negative")
            if args.m_forced_operator_best_of < 1:
                raise ValueError(
                    "forced M operator best-of must be positive")
            if (args.m_forced_operator_best_of > 1
                    and args.m_forced_operator != "replace"):
                raise ValueError(
                    "forced M operator best-of requires replace")
            if (args.m_forced_operator_best_of > 1
                    and (args.m_forced_operator_positive_probe
                         or args.m_forced_operator_reversible_probe)):
                raise ValueError(
                    "forced M operator best-of cannot be combined with probes")
            if (args.m_forced_replace_selector != "base_random"
                    and args.m_forced_operator != "replace"):
                raise ValueError(
                    "directed replace selector requires forced replace")
            if (args.m_forced_replace_selector != "base_random"
                    and (args.m_forced_operator_best_of > 1
                         or args.m_forced_operator_positive_probe
                         or args.m_forced_operator_reversible_probe)):
                raise ValueError(
                    "directed replace selector cannot be combined with "
                    "best-of or probes")
            if args.m_forced_operator_probe_max_attempts > 1:
                if not args.m_forced_operator_reversible_probe:
                    raise ValueError(
                        "repeated operator probes require reversible probe")
                if args.m_forced_operator_probe_retry_events < 1:
                    raise ValueError(
                        "repeated operator probes require a positive retry "
                        "interval")
        elif (args.m_forced_operator_min_generation is not None
              or args.m_forced_operator_positive_probe
              or args.m_forced_operator_reversible_probe
              or args.m_forced_operator_probe_max_attempts != 1
              or args.m_forced_operator_probe_retry_events != 0
              or args.m_forced_operator_best_of != 1
              or args.m_forced_replace_selector != "base_random"):
            raise ValueError(
                "forced M operator probe/schedule requires "
                "--m-forced-operator")
        if ((args.m_continuation_event is None)
                != (args.m_continuation_seed is None)):
            raise ValueError(
                "M continuation event and seed must be supplied together")
        if args.m_continuation_event is not None:
            if args.component != "M":
                raise ValueError("M continuation replay requires component M")
            if args.m_continuation_event < 1:
                raise ValueError(
                    "M continuation event must be a positive integer")
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
                    args.inference_policy_seed + "ICM".index(name) + 1,
                    args.inference_group_margin)
                for name in args.component
            }
        else:
            trainer = _load(
                args.checkpoint, args.component, args.device,
                args.inference_action_mode, args.inference_policy_seed,
                args.inference_group_margin)
        summary = run_episode(
            args.data, args.instance, args.out, trainer, normalizer,
            algorithm_seed=args.algorithm_seed,
            evaluation_budget=args.evaluation_budget, pop_size=args.pop,
            generations=args.gens, reward_config=RewardConfig(),
            evaluation_mode=args.evaluation_mode, component=args.component,
            scenario_count=args.scenario_count,
            ccp_confidence=args.ccp_confidence,
            uncertainty_profile=args.uncertainty_profile,
            rule_initialization_normalization=(
                rule_normalization if args.component == "RULE-I" else None),
            rule_initialization_profile=args.initialization_rule_profile,
            initialization_learning_rule_normalization=(
                rule_normalization
                if args.initialization_rule_top_k is not None else None),
            initialization_learning_rule_top_k=args.initialization_rule_top_k,
            model_choice_event=args.model_choice_event,
            i_forced_candidate_index=args.i_forced_candidate_index,
            c_forced_joint_candidate_index=(
                args.c_forced_joint_candidate_index),
            m_forced_candidate_index=args.m_forced_candidate_index,
            m_forced_operator=args.m_forced_operator,
            m_forced_operator_min_generation=(
                args.m_forced_operator_min_generation),
            m_forced_operator_probability=(
                args.m_forced_operator_probability),
            m_forced_operator_positive_probe=(
                args.m_forced_operator_positive_probe),
            m_forced_operator_reversible_probe=(
                args.m_forced_operator_reversible_probe),
            m_forced_operator_probe_max_attempts=(
                args.m_forced_operator_probe_max_attempts),
            m_forced_operator_probe_retry_events=(
                args.m_forced_operator_probe_retry_events),
            m_forced_operator_best_of=(
                args.m_forced_operator_best_of),
            m_forced_replace_selector=(
                args.m_forced_replace_selector),
            m_continuation_event=args.m_continuation_event,
            m_continuation_seed=args.m_continuation_seed,
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
