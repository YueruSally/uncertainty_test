"""One deterministic or CCP100 PPO training/frozen-policy episode."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import time

import numpy as np

import baseline_uncertainty as base
from multi_instance_logging import fingerprint, write_json
from run_ev_ccp_oos_pilot import candidate_rows, json_candidate, scenario_digest

from .archive import RewardArchive, RewardConfig, exact_hv_3d
from .comparison_controllers import BaseBudgetController, CombinedPPOController
from .controller import PPOController
from .crossover_controller import CrossoverPPOController
from .initialization_controller import InitializationPPOController
from .initialization_rule import InitializationRuleController
from .initialization_rule import INITIALIZATION_RULE_PROFILES
from .initialization_rule_features import validate_rule_normalization
from .instance_catalog import read_instance
from .graph_features import build_graph_observation
from .v2_controls import validate_model_choice_probability


FINAL_OBJECTIVE_ORDER = ("cost", "emission", "makespan")

CCP_UNCERTAINTY_PROFILES = {
    "legacy": {
        "default_scenario_count": 100,
        "mode_time_cv": {"road": 0.15, "rail": 0.10, "water": 0.20},
        "mode_time_max_factor": {"road": 2.0, "rail": 1.8, "water": 2.5},
        "border_delay_cv": {"road": 0.45, "rail": 0.60, "water": 0.40},
        "border_delay_max_factor": 3.5,
        "border_mean_source": "directional_csv",
        "border_include_customs": True,
    },
    "transport_border_saa_v1": {
        "default_scenario_count": 200,
        "mode_time_cv": {"road": 0.15, "rail": 0.10, "water": 0.20},
        "mode_time_max_factor": {"road": None, "rail": None, "water": None},
        "border_delay_cv": {"road": 0.50, "rail": 0.50, "water": 0.50},
        "border_delay_max_factor": 3.5,
        "border_mean_source": "node_border",
        "border_include_customs": False,
    },
}


def resolve_uncertainty_profile(
        evaluation_mode, uncertainty_profile="legacy", scenario_count=None):
    """Resolve one explicit frozen-SAA uncertainty configuration."""
    if evaluation_mode not in ("deterministic", "ccp100", "ccp"):
        raise ValueError(
            "evaluation_mode must be 'deterministic', 'ccp100', or 'ccp'")
    try:
        profile = dict(CCP_UNCERTAINTY_PROFILES[uncertainty_profile])
    except KeyError as error:
        raise ValueError(
            f"unknown uncertainty profile: {uncertainty_profile}") from error
    if evaluation_mode == "deterministic":
        resolved_count = 1
    elif scenario_count is None:
        resolved_count = (
            100 if evaluation_mode == "ccp100" else
            int(profile["default_scenario_count"]))
    else:
        resolved_count = int(scenario_count)
    if resolved_count < 1:
        raise ValueError("scenario_count must be positive")
    profile["name"] = uncertainty_profile
    profile["scenario_count"] = resolved_count
    return profile


def resolve_combined_graph_representation(policy_representations):
    """Map component-specific candidate schemas to one shared graph schema."""
    families = {
        ("legacy" if representation == "legacy" else "ordered-path")
        for representation in policy_representations.values()}
    if len(families) > 1:
        raise ValueError(
            "combined checkpoints must use one common graph representation family")
    return "path_v2" if families == {"ordered-path"} else "legacy"


def _scenario_sampling_diagnostics(scenarios, arcs):
    """Summarise the realised frozen draws without changing optimisation."""
    result = {"transport_multiplier": {}, "border_multiplier": {}}
    arc_modes = {
        (arc.from_node, arc.to_node, arc.mode): arc.mode for arc in arcs}
    for mode in ("road", "rail", "water"):
        arrays = [
            np.asarray(values, dtype=float)
            for key, values in scenarios.travel_multiplier.items()
            if arc_modes.get(key) == mode
        ]
        if arrays:
            values = np.concatenate(arrays)
            mean = float(np.mean(values))
            result["transport_multiplier"][mode] = {
                "sample_count": int(values.size),
                "mean": mean,
                "cv": (float(np.std(values)) / mean if mean > 0.0 else 0.0),
                "maximum": float(np.max(values)),
            }
    for mode in ("road", "rail", "water"):
        arrays = []
        for key, values in scenarios.border_delay_h.items():
            mean_h = float(scenarios.border_event_mean_h.get(key, 0.0))
            if key[2] == mode and mean_h > 0.0:
                arrays.append(np.asarray(values, dtype=float) / mean_h)
        if arrays:
            values = np.concatenate(arrays)
            mean = float(np.mean(values))
            result["border_multiplier"][mode] = {
                "sample_count": int(values.size),
                "mean": mean,
                "cv": (float(np.std(values)) / mean if mean > 0.0 else 0.0),
                "maximum": float(np.max(values)),
            }
    return result


def _front_uncertainty_diagnostics(front):
    result = {"candidate_count": len(front), "objectives": {}}
    for name, attribute in (
            ("cost", "cost_s"), ("emission", "emission_s"),
            ("makespan", "makespan_s")):
        standard_deviations = [
            float(np.std(np.asarray(getattr(individual, attribute), dtype=float)))
            for individual in front
            if getattr(individual, attribute, None) is not None
        ]
        result["objectives"][name] = {
            "candidates_with_scenarios": len(standard_deviations),
            "mean_scenario_standard_deviation": (
                float(np.mean(standard_deviations))
                if standard_deviations else None),
            "maximum_scenario_standard_deviation": (
                float(np.max(standard_deviations))
                if standard_deviations else None),
        }
    return result


def _optimisation_vector(row):
    """Return stored final objectives in the normalizer's fixed axis order."""
    objectives = row["optimisation_objectives"]
    if not isinstance(objectives, dict):
        raise TypeError("optimisation_objectives must be an objective-name mapping")
    return tuple(float(objectives[name]) for name in FINAL_OBJECTIVE_ORDER)


def run_episode(data: Path, instance_path: Path, out: Path, trainer, normalizer,
                algorithm_seed: int, evaluation_budget=15_000,
                pop_size=100, generations=1_000, reward_config=None,
                evaluation_mode="ccp100", component="M",
                rule_initialization_normalization=None,
                rule_initialization_profile="legacy",
                initialization_learning_rule_normalization=None,
                initialization_learning_rule_top_k=None,
                model_choice_probabilities=None,
                model_choice_event=None,
                i_forced_candidate_index=None,
                c_forced_joint_candidate_index=None,
                m_forced_candidate_index=None,
                m_forced_operator=None,
                m_forced_operator_min_generation=None,
                m_forced_operator_probability=1.0,
                m_forced_operator_positive_probe=False,
                m_forced_operator_reversible_probe=False,
                m_forced_operator_probe_max_attempts=1,
                m_forced_operator_probe_retry_events=0,
                m_forced_operator_best_of=1,
                m_forced_replace_selector="base_random",
                m_continuation_event=None,
                m_continuation_seed=None,
                learning_protocol_version=None,
                candidate_audit=None,
                scenario_count=None,
                ccp_confidence=0.90,
                uncertainty_profile="legacy",
                crossover_reward_mode="archive_hv"):
    if pop_size < 2 or pop_size % 2 or generations < 1:
        raise ValueError("pop_size must be even and >=2; generations must be positive")
    if evaluation_budget is not None and evaluation_budget < pop_size + 4:
        raise ValueError("evaluation_budget must be at least pop_size+4")
    resolved_uncertainty = resolve_uncertainty_profile(
        evaluation_mode, uncertainty_profile, scenario_count)
    ccp_confidence = float(ccp_confidence)
    if not 0.0 < ccp_confidence <= 1.0:
        raise ValueError("ccp_confidence must be in (0, 1]")
    component = str(component).upper()
    if crossover_reward_mode not in (
            "archive_hv", "paired_base_relative",
            "paired_archive_then_quality"):
        raise ValueError(
            "crossover_reward_mode must be archive_hv, paired_base_relative, "
            "or paired_archive_then_quality")
    if crossover_reward_mode != "archive_hv" and component != "C":
        raise ValueError(
            "paired crossover reward currently requires component C")
    combined_components = {
        "IC": {"I", "C"},
        "IM": {"I", "M"},
        "CM": {"C", "M"},
        "ICM": {"I", "C", "M"},
    }
    if component not in ("M", "C", "I", *combined_components, "BASE", "RULE-I"):
        raise ValueError(
            "component must be a single/combined learning component, BASE, or RULE-I")
    if component in ("M", "C", "I") and trainer.component != component:
        raise ValueError("trainer component does not match episode component")
    if component in combined_components and (
            not isinstance(trainer, dict)
            or set(trainer) != combined_components[component]
            or any(item.component != name for name, item in trainer.items())):
        raise ValueError(f"{component} requires matching component trainers")
    if component in ("BASE", "RULE-I") and trainer is not None:
        raise ValueError(f"{component} does not accept a policy trainer")
    episode_trainers = (
        tuple(trainer.values()) if component in combined_components else
        () if trainer is None else (trainer,))
    policy_candidate_representations = {
        item.component: getattr(
            getattr(item, "config", None), "candidate_representation", "legacy")
        for item in episode_trainers}
    candidate_representation = resolve_combined_graph_representation(
        policy_candidate_representations)
    rule_initialization_profile = str(rule_initialization_profile)
    if rule_initialization_profile not in INITIALIZATION_RULE_PROFILES:
        raise ValueError(
            "invalid initialization rule profile: "
            f"{rule_initialization_profile}")
    if (component == "RULE-I"
            and rule_initialization_profile == "legacy"
            and rule_initialization_normalization is None):
        raise ValueError("legacy RULE-I requires a path-feature normalization")
    if (component != "RULE-I"
            and rule_initialization_profile != "legacy"):
        raise ValueError("directed initialization profile requires RULE-I")
    if learning_protocol_version not in (None, 1, 2):
        raise ValueError("learning protocol version must be 1 or 2")
    audit = dict(candidate_audit or {})
    allowed_audit = {"states", "candidates", "replicates"}
    if set(audit) - allowed_audit:
        raise ValueError("unknown candidate audit options")
    audit = {
        "states": int(audit.get("states", 0)),
        "candidates": int(audit.get("candidates", 4)),
        "replicates": int(audit.get("replicates", 1)),
    }
    if audit["states"] < 0 or audit["candidates"] < 2 or audit["replicates"] < 1:
        raise ValueError("invalid candidate audit dimensions")
    if audit["states"] and component != "M":
        raise ValueError("fixed-state candidate audit currently supports only M")
    if ((initialization_learning_rule_normalization is None)
            != (initialization_learning_rule_top_k is None)):
        raise ValueError(
            "learned I rule normalization and top-k must be supplied together")
    if initialization_learning_rule_top_k is not None:
        if component != "I" and "I" not in combined_components.get(component, set()):
            raise ValueError("I rule prefilter requires an active I component")
        if int(initialization_learning_rule_top_k) < 2:
            raise ValueError("learned I rule top-k must be at least 2")
        validate_rule_normalization(
            initialization_learning_rule_normalization)
    probabilities = {"C": 1.0, "M": 1.0}
    supplied_probabilities = dict(model_choice_probabilities or {})
    if set(supplied_probabilities) - {"C", "M"}:
        raise ValueError("model choice probabilities accept only C and M")
    probabilities.update({
        name: validate_model_choice_probability(value)
        for name, value in supplied_probabilities.items()
    })
    allowed_probability_components = (
        {component} if component in ("C", "M") else
        combined_components.get(component, set()) & {"C", "M"})
    if any(name not in allowed_probability_components
           for name in supplied_probabilities):
        raise ValueError(
            "model choice probabilities do not match the episode component")
    if model_choice_event is not None:
        model_choice_event = int(model_choice_event)
        if model_choice_event < 1:
            raise ValueError("model choice event must be a positive integer")
        if component not in ("I", "C", "M"):
            raise ValueError("model choice event supports only I, C, or M")
        if component in probabilities and probabilities[component] != 1.0:
            raise ValueError(
                "model choice event cannot be combined with probability gating")
    if i_forced_candidate_index is not None:
        i_forced_candidate_index = int(i_forced_candidate_index)
        if i_forced_candidate_index < 0:
            raise ValueError("forced I candidate index cannot be negative")
        if component != "I" or model_choice_event is None:
            raise ValueError(
                "forced I candidate index requires component I and a "
                "model choice event")
    if c_forced_joint_candidate_index is not None:
        c_forced_joint_candidate_index = int(c_forced_joint_candidate_index)
        if c_forced_joint_candidate_index < 0:
            raise ValueError("forced C joint candidate index cannot be negative")
        if component != "C" or model_choice_event is None:
            raise ValueError(
                "forced C joint candidate index requires component C and a "
                "model choice event")
    if m_forced_candidate_index is not None:
        m_forced_candidate_index = int(m_forced_candidate_index)
        if m_forced_candidate_index < 0:
            raise ValueError("forced M candidate index cannot be negative")
        if component != "M" or model_choice_event is None:
            raise ValueError(
                "forced M candidate index requires component M and a "
                "model choice event")
    if m_forced_operator is not None:
        m_forced_operator = str(m_forced_operator)
        if m_forced_operator not in ("add", "del", "mod", "mode", "replace"):
            raise ValueError("invalid forced M operator")
        if component != "M":
            raise ValueError(
                "forced M operator requires component M")
        if ((model_choice_event is None)
                == (m_forced_operator_min_generation is None)):
            raise ValueError(
                "forced M operator requires exactly one of a model choice "
                "event and a minimum generation schedule")
        if m_forced_candidate_index is not None:
            raise ValueError(
                "forced M operator and candidate index are mutually exclusive")
        m_forced_operator_probability = float(m_forced_operator_probability)
        if not 0.0 <= m_forced_operator_probability <= 1.0:
            raise ValueError(
                "forced M operator probability must be in [0, 1]")
        m_forced_operator_positive_probe = bool(
            m_forced_operator_positive_probe)
        m_forced_operator_reversible_probe = bool(
            m_forced_operator_reversible_probe)
        m_forced_operator_probe_max_attempts = int(
            m_forced_operator_probe_max_attempts)
        m_forced_operator_probe_retry_events = int(
            m_forced_operator_probe_retry_events)
        m_forced_operator_best_of = int(m_forced_operator_best_of)
        m_forced_replace_selector = str(m_forced_replace_selector)
        if m_forced_replace_selector not in (
                "base_random", "joint_extreme_cycle", "joint_balanced",
                "joint_pareto_cycle"):
            raise ValueError("invalid forced M replace selector")
        if (m_forced_operator_positive_probe
                and m_forced_operator_probability != 1.0):
            raise ValueError(
                "positive operator probe requires probability 1")
        if (m_forced_operator_reversible_probe
                and not m_forced_operator_positive_probe):
            raise ValueError(
                "reversible operator probe requires positive operator probe")
        if m_forced_operator_probe_max_attempts < 1:
            raise ValueError(
                "operator probe maximum attempts must be positive")
        if m_forced_operator_probe_retry_events < 0:
            raise ValueError(
                "operator probe retry events cannot be negative")
        if m_forced_operator_best_of < 1:
            raise ValueError("forced M operator best-of must be positive")
        if (m_forced_operator_best_of > 1
                and m_forced_operator != "replace"):
            raise ValueError("forced M operator best-of requires replace")
        if (m_forced_operator_best_of > 1
                and (m_forced_operator_positive_probe
                     or m_forced_operator_reversible_probe)):
            raise ValueError(
                "forced M operator best-of cannot be combined with probes")
        if (m_forced_replace_selector != "base_random"
                and m_forced_operator != "replace"):
            raise ValueError(
                "directed replace selector requires forced replace")
        if (m_forced_replace_selector != "base_random"
                and (m_forced_operator_best_of > 1
                     or m_forced_operator_positive_probe
                     or m_forced_operator_reversible_probe)):
            raise ValueError(
                "directed replace selector cannot be combined with best-of "
                "or probes")
        if m_forced_operator_probe_max_attempts > 1:
            if not m_forced_operator_reversible_probe:
                raise ValueError(
                    "repeated operator probes require a reversible probe")
            if m_forced_operator_probe_retry_events < 1:
                raise ValueError(
                    "repeated operator probes require a positive retry interval")
        if m_forced_operator_min_generation is not None:
            m_forced_operator_min_generation = int(
                m_forced_operator_min_generation)
            if m_forced_operator_min_generation < 0:
                raise ValueError(
                    "forced M operator minimum generation cannot be negative")
    elif (m_forced_operator_min_generation is not None
          or m_forced_operator_positive_probe
          or m_forced_operator_reversible_probe
          or m_forced_operator_probe_max_attempts != 1
          or m_forced_operator_probe_retry_events != 0
          or m_forced_operator_best_of != 1
          or m_forced_replace_selector != "base_random"):
        raise ValueError(
            "forced M operator probe/schedule requires a forced operator")
    if (m_continuation_event is None) != (m_continuation_seed is None):
        raise ValueError(
            "M continuation event and seed must be supplied together")
    if m_continuation_event is not None:
        m_continuation_event = int(m_continuation_event)
        m_continuation_seed = int(m_continuation_seed)
        if component != "M":
            raise ValueError("M continuation replay requires component M")
        if m_continuation_event < 1:
            raise ValueError("M continuation event must be a positive integer")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"episode output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    instance, batches = read_instance(instance_path)
    if hashlib.sha256(data.read_bytes()).hexdigest() != instance["source_data_sha256"]:
        raise ValueError("network workbook changed after catalog generation")
    started = time.perf_counter()
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    (nodes, regions, hold, proc, trans_cost, arcs, timetables, _raw,
     wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
     border_delay, theta, _) = network
    active_border_delay = (
        base.load_border_delay_map_from_workbook(
            data,
            include_customs=resolved_uncertainty["border_include_customs"])
        if resolved_uncertainty["border_mean_source"] == "node_border"
        else border_delay)
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    random.seed(instance["path_seed"])
    np.random.seed(instance["path_seed"] % (2**32 - 1))
    paths = base.build_path_library(nodes, regions, arcs, batches, timetable, lookup)
    base.sanity_check_path_lib(batches, paths)
    if evaluation_mode == "deterministic":
        scenario_seed = 0
        scenarios = base.build_scenario_set(
            arcs, active_border_delay, 1, scenario_seed, stochastic=False,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            border_mean_source=resolved_uncertainty["border_mean_source"])
        option_scenarios = scenarios
        option_mode = "deterministic"
        risk_metric = "deterministic"
    else:
        scenario_seed = instance["ccp100_seed"]
        scenarios = base.build_scenario_set(
            arcs, active_border_delay, resolved_uncertainty["scenario_count"],
            scenario_seed, stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            mode_time_cv=resolved_uncertainty["mode_time_cv"],
            mode_time_max_factor=(
                resolved_uncertainty["mode_time_max_factor"]),
            border_delay_cv=resolved_uncertainty["border_delay_cv"],
            border_delay_max_factor=(
                resolved_uncertainty["border_delay_max_factor"]),
            border_mean_source=resolved_uncertainty["border_mean_source"])
        option_scenarios = base.build_expected_value_scenario_set(
            arcs, active_border_delay, seed=0,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS,
            border_mean_source=resolved_uncertainty["border_mean_source"])
        option_mode = "ev"
        risk_metric = "ccp"
    options = base.build_reliable_path_options(
        batches, paths, timetable, trans_map, active_border_delay,
        option_scenarios, mode=option_mode)
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = risk_metric
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = \
        ccp_confidence
    random.seed(algorithm_seed)
    np.random.seed(algorithm_seed % (2**32 - 1))
    seeds = {"algorithm": algorithm_seed, "path": instance["path_seed"],
             "training": scenario_seed}
    objective_divisor = (
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, sum(float(batch.quantity) for batch in batches)),
        max(1.0, max(float(batch.LT) for batch in batches)
            - min(float(batch.ET) for batch in batches)),
    )
    episode_normalizer = normalizer.with_divisor(objective_divisor)
    graph_observation = build_graph_observation(
        nodes, regions, hold, proc, trans_cost, arcs, batches, timetables,
        candidate_representation)
    method = (
        "base-nsga2" if component == "BASE" else
        "rule-initialization" if component == "RULE-I" else
        f"transformer-ppo-{'+'.join(component.lower())}-frozen"
        if component in combined_components else
        trainer.name)
    policy_training = (
        False if component in ("BASE", "RULE-I") else
        any(item.training for item in trainer.values())
        if component in combined_components else
        bool(trainer.training))
    policy_checkpoint = (
        None if component in ("BASE", "RULE-I") else
        {name: item.checkpoint_provenance for name, item in trainer.items()}
        if component in combined_components else trainer.checkpoint_provenance)
    def inference_metadata(item):
        return {
            "action_mode": (
                "sample" if item.training else
                getattr(item, "inference_action_mode", "argmax")),
            "policy_seed": getattr(item, "policy_seed", None),
            "grouped_argmax_margin": float(getattr(
                item, "grouped_argmax_margin", 0.0)),
        }

    policy_inference = (
        None if component in ("BASE", "RULE-I") else
        {name: inference_metadata(item) for name, item in trainer.items()}
        if component in combined_components else inference_metadata(trainer))
    policy_trainers = (
        trainer.values() if component in combined_components else
        () if trainer is None else (trainer,))
    checkpoint_v2 = any(
        ((item.checkpoint_provenance or {}).get("learning_control") or {}).get(
            "protocol") == "nsga2-learning-components-v2"
        for item in policy_trainers
    )
    derived_v2 = (
        initialization_learning_rule_top_k is not None
        or any(probabilities[name] != 1.0
               for name in allowed_probability_components)
        or model_choice_event is not None
        or checkpoint_v2
    )
    v2_active = (
        derived_v2 if learning_protocol_version is None
        else learning_protocol_version == 2)
    config = {
        "schema_version": 3, "method": method,
        "protocol": (
            "nsga2-learning-components-v2"
            if v2_active else "nsga2-learning-components-v1"),
        "component": component,
        "evaluation_mode": evaluation_mode,
        "instance_id": instance["instance_id"], "configuration": instance["configuration"],
        "instance_digest": instance["instance_digest"], "K": instance["K"],
        "deadline_window_alpha": instance["deadline_window_alpha"],
        "population": pop_size, "generations": generations,
        "alpha": ccp_confidence,
        "evaluation_budget": evaluation_budget,
        "training_size": scenarios.size,
        "scenario_digest": scenario_digest(scenarios), "seeds": seeds,
        "uncertainty": {
            **resolved_uncertainty,
            "distribution": "lognormal",
            "transport_arc_independence": True,
            "border_event_independence": True,
            "transport_border_independence": True,
            "frozen_saa": True,
            "sampling_diagnostics": _scenario_sampling_diagnostics(
                scenarios, arcs),
        },
        "operator_probabilities": dict(zip(base.OPS, base._FIXED_OP_PROBS)),
        "crossover_rate": base.CROSSOVER_RATE, "mutation_rate": base.MUTATION_RATE,
        "external_archive_role": "reward-only; excluded from environmental selection",
        "reward_definition": (
            "signed same-state archive HV difference with standalone-set HV "
            "tie-break"
            if crossover_reward_mode == "paired_archive_then_quality" else
            "signed same-state selected-minus-reference pre-mutation archive HV"
            if crossover_reward_mode == "paired_base_relative" else
            "scaled positive joint archive HV increment only"),
        "crossover_reward_mode": crossover_reward_mode,
        "counterfactual_training_evaluations_excluded_from_search_budget": (
            crossover_reward_mode in (
                "paired_base_relative", "paired_archive_then_quality")),
        "reward_clipping": False,
        "intermediate_post_crossover_evaluation": (
            crossover_reward_mode in (
                "paired_base_relative", "paired_archive_then_quality")),
        "policy_training": policy_training,
        "policy_checkpoint": policy_checkpoint,
        "policy_inference": policy_inference,
        "policy_candidate_representations": (
            policy_candidate_representations or None),
        "initialization_rule": (
            rule_initialization_normalization if component == "RULE-I" else None),
        "initialization_rule_profile": (
            rule_initialization_profile if component == "RULE-I" else None),
        "learning_control": {
            "version": 2 if v2_active else 1,
            "initialization_rule_prefilter": (
                None if initialization_learning_rule_top_k is None else {
                    "top_k": int(initialization_learning_rule_top_k),
                    "normalization_digest": (
                        initialization_learning_rule_normalization.get(
                            "normalization_digest")),
                }),
            "model_choice_probabilities": {
                name: probabilities[name]
                for name in sorted(allowed_probability_components)
            },
            "model_choice_event": model_choice_event,
            "i_forced_candidate_index": i_forced_candidate_index,
            "c_forced_joint_candidate_index": c_forced_joint_candidate_index,
            "m_forced_candidate_index": m_forced_candidate_index,
            "m_forced_operator": m_forced_operator,
            "m_forced_operator_min_generation": (
                m_forced_operator_min_generation),
            "m_forced_operator_probability": m_forced_operator_probability,
            "m_forced_operator_positive_probe": (
                m_forced_operator_positive_probe),
            "m_forced_operator_reversible_probe": (
                m_forced_operator_reversible_probe),
            "m_forced_operator_probe_max_attempts": (
                m_forced_operator_probe_max_attempts),
            "m_forced_operator_probe_retry_events": (
                m_forced_operator_probe_retry_events),
            "m_forced_operator_best_of": m_forced_operator_best_of,
            "m_forced_replace_selector": m_forced_replace_selector,
            "m_continuation_event": m_continuation_event,
            "m_continuation_seed": m_continuation_seed,
            "control_random_stream": "private-sha256-seeded",
        },
        "candidate_audit": (
            audit if audit["states"] else None),
        "network_encoding": graph_observation["schema"],
        "normalization": {"minimum": episode_normalizer.minimum,
                          "maximum": episode_normalizer.maximum,
                          "reference": episode_normalizer.reference,
                          "objective_divisor": objective_divisor,
                          "definition": "cost/total_teu, emission/total_teu, makespan/instance_horizon"},
    }
    write_json(out / "configuration.json", config)
    archive = RewardArchive(episode_normalizer)
    controller_types = {
        "M": PPOController,
        "C": CrossoverPPOController,
        "I": InitializationPPOController,
    }
    if component == "BASE":
        controller = BaseBudgetController(
            out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
            method=method, instance=instance, seeds=seeds,
            normalizer=episode_normalizer)
    elif component == "RULE-I":
        controller = InitializationRuleController(
            out, instance["instance_id"], config["scenario_digest"],
            evaluation_budget, method=method, instance=instance, seeds=seeds,
            normalizer=episode_normalizer,
            rule_normalization=rule_initialization_normalization,
            reliable_options=options,
            profile=rule_initialization_profile)
    elif component in combined_components:
        controller = CombinedPPOController(
            out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
            trainers=trainer, normalizer=episode_normalizer,
            reward_config=reward_config or RewardConfig(), batches=batches,
            generations=generations, pop_size=pop_size,
            graph_observation=graph_observation, method=method,
            instance=instance, seeds=seeds,
            initialization_rule_normalization=(
                initialization_learning_rule_normalization),
            initialization_rule_top_k=initialization_learning_rule_top_k,
            model_choice_probabilities=probabilities)
    else:
        controller_type = controller_types[component]
        controller_options = {}
        if component == "I":
            controller_options.update(
                rule_normalization=initialization_learning_rule_normalization,
                rule_top_k=initialization_learning_rule_top_k,
                model_choice_event=model_choice_event,
                forced_candidate_index=i_forced_candidate_index)
        elif component in ("C", "M"):
            controller_options["model_choice_probability"] = probabilities[component]
            if component == "C":
                controller_options["crossover_reward_mode"] = (
                    crossover_reward_mode)
                controller_options["forced_joint_candidate_index"] = (
                    c_forced_joint_candidate_index)
            if component == "M":
                controller_options["forced_candidate_index"] = (
                    m_forced_candidate_index)
                controller_options["forced_operator"] = m_forced_operator
                controller_options["forced_operator_min_generation"] = (
                    m_forced_operator_min_generation)
                controller_options["forced_operator_probability"] = (
                    m_forced_operator_probability)
                controller_options["forced_operator_positive_probe"] = (
                    m_forced_operator_positive_probe)
                controller_options["forced_operator_reversible_probe"] = (
                    m_forced_operator_reversible_probe)
                controller_options["forced_operator_probe_max_attempts"] = (
                    m_forced_operator_probe_max_attempts)
                controller_options["forced_operator_probe_retry_events"] = (
                    m_forced_operator_probe_retry_events)
                controller_options["forced_operator_best_of"] = (
                    m_forced_operator_best_of)
                controller_options["forced_replace_selector"] = (
                    m_forced_replace_selector)
                controller_options["continuation_event"] = (
                    m_continuation_event)
                controller_options["continuation_seed"] = (
                    m_continuation_seed)
            if component in ("C", "M"):
                controller_options["model_choice_event"] = model_choice_event
            if component == "M" and audit["states"]:
                controller_options.update(
                    candidate_audit_states=audit["states"],
                    candidate_audit_candidates=audit["candidates"],
                    candidate_audit_replicates=audit["replicates"],
                )
        controller = controller_type(
            out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
            trainer=trainer, archive=archive,
            reward_config=reward_config or RewardConfig(), batches=batches,
            generations=generations, pop_size=pop_size, method=method,
            instance=instance, seeds=seeds, graph_observation=graph_observation,
            **controller_options)
    base.ACTIVE_MUTATION_LOGGER = controller
    try:
        population = base.run_nsga2(
            nodes, regions, hold, proc, trans_cost, arcs, timetables, batches,
            wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
            active_border_delay, theta, paths, options, pop_size=pop_size,
            generations=generations)[0]
        controller.flush_events(population)
        controller.finish_episode()
    finally:
        base.ACTIVE_MUTATION_LOGGER = None
        controller.close()
    fronts = base.fast_non_dominated_sort(population)
    front = [population[index] for index in fronts[0] if population[index].feasible]
    rows = candidate_rows(method, instance["instance_id"], front, config)
    final_points = [
        episode_normalizer.transform(_optimisation_vector(row))
        for row in rows
    ]
    final_hypervolume = exact_hv_3d(
        final_points, episode_normalizer.reference)
    write_json(out / "final_feasible_nondominated.json", [json_candidate(row) for row in rows])
    write_json(out / "final_archive_lineage.json", [{
        "source_solution_id": row["source_solution_id"],
        "decision_fingerprint": fingerprint(row["individual"]),
        "optimisation_objectives": row["optimisation_objectives"],
    } for row in rows])
    optimisation_seconds = time.perf_counter() - started
    measured_seconds = (
        controller.objective_evaluation_seconds
        + float(getattr(
            controller, "counterfactual_evaluation_seconds", 0.0))
        + controller.feature_construction_seconds
        + controller.inference_seconds
        + controller.ppo_update_seconds
        + float(getattr(controller, "candidate_audit_seconds", 0.0)))
    summary = {
        "completed": True, "method": method,
        "real_objective_evaluation_count": controller.evaluations,
        "ccp_evaluation_count": controller.evaluations,
        "component": component,
        "component_attempts": getattr(controller, "total_attempts", 0),
        "effective_component_modifications": getattr(
            controller, "total_effective", 0),
        "no_eligible_candidate": getattr(controller, "no_candidate", 0),
        "inference_seconds": controller.inference_seconds,
        "feature_construction_seconds": controller.feature_construction_seconds,
        "objective_evaluation_seconds": controller.objective_evaluation_seconds,
        "ppo_update_seconds": controller.ppo_update_seconds,
        "other_optimisation_seconds": max(0.0, optimisation_seconds - measured_seconds),
        "optimisation_seconds": optimisation_seconds,
        "stop_reason": (
            "evaluation_budget"
            if not controller.has_budget(
                controller.budget_reserve("offspring_pair"))
            else "generation_limit"),
        "evaluation_budget": evaluation_budget,
        "generations_completed": controller.generations_completed,
        "final_front_size": len(rows),
        "final_hypervolume": final_hypervolume,
        "final_front_uncertainty": _front_uncertainty_diagnostics(
            [row["individual"] for row in rows]),
        **controller.summary(),
    }
    if component == "M":
        summary["mutation_attempts"] = controller.total_attempts
        summary["effective_modifications"] = controller.total_effective
    elif component == "C":
        summary["crossover_attempts"] = controller.total_attempts
        summary["effective_crossovers"] = controller.total_effective
    elif component == "I":
        summary["initialization_attempts"] = controller.total_attempts
        summary["initializations_different_from_shadow_base"] = controller.total_effective
    elif component == "RULE-I":
        summary["initialization_attempts"] = controller.total_attempts
        summary["initializations_different_from_shadow_base"] = controller.total_effective
    elif component in combined_components:
        summary["component_attempts"] = {
            name: item.total_attempts for name, item in controller.controllers.items()}
        summary["effective_component_modifications"] = {
            name: item.total_effective for name, item in controller.controllers.items()}
        summary["no_eligible_candidate"] = {
            name: item.no_candidate for name, item in controller.controllers.items()}
    write_json(out / "COMPLETE.json", summary)
    return summary
