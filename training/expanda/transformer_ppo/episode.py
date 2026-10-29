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
from .instance_catalog import read_instance
from .graph_features import build_graph_observation


FINAL_OBJECTIVE_ORDER = ("cost", "emission", "makespan")


def _optimisation_vector(row):
    """Return stored final objectives in the normalizer's fixed axis order."""
    objectives = row["optimisation_objectives"]
    if not isinstance(objectives, dict):
        raise TypeError("optimisation_objectives must be an objective-name mapping")
    return tuple(float(objectives[name]) for name in FINAL_OBJECTIVE_ORDER)


def run_episode(data: Path, instance_path: Path, out: Path, trainer, normalizer,
                algorithm_seed: int, evaluation_budget=15_000,
                pop_size=100, generations=1_000, reward_config=None,
                evaluation_mode="ccp100", component="M"):
    if pop_size < 2 or pop_size % 2 or generations < 1:
        raise ValueError("pop_size must be even and >=2; generations must be positive")
    if evaluation_budget is not None and evaluation_budget < pop_size + 4:
        raise ValueError("evaluation_budget must be at least pop_size+4")
    if evaluation_mode not in ("deterministic", "ccp100"):
        raise ValueError("evaluation_mode must be 'deterministic' or 'ccp100'")
    component = str(component).upper()
    if component not in ("M", "C", "I", "ICM", "BASE"):
        raise ValueError("component must be 'M', 'C', 'I', 'ICM', or 'BASE'")
    if component in ("M", "C", "I") and trainer.component != component:
        raise ValueError("trainer component does not match episode component")
    if component == "ICM" and (
            not isinstance(trainer, dict) or set(trainer) != {"I", "C", "M"}
            or any(item.component != name for name, item in trainer.items())):
        raise ValueError("ICM requires matching I, C, and M trainers")
    if component == "BASE" and trainer is not None:
        raise ValueError("BASE does not accept a policy trainer")
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
            arcs, border_delay, 1, scenario_seed, stochastic=False,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        option_scenarios = scenarios
        option_mode = "deterministic"
        risk_metric = "deterministic"
    else:
        scenario_seed = instance["ccp100_seed"]
        scenarios = base.build_scenario_set(
            arcs, border_delay, 100, scenario_seed, stochastic=True,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        option_scenarios = base.build_expected_value_scenario_set(
            arcs, border_delay, seed=0,
            border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
        option_mode = "ev"
        risk_metric = "ccp"
    options = base.build_reliable_path_options(
        batches, paths, timetable, trans_map, border_delay,
        option_scenarios, mode=option_mode)
    base.ACTIVE_SCENARIO_SET = scenarios
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = risk_metric
    base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = 0.90
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
        nodes, regions, hold, proc, trans_cost, arcs, batches)
    method = (
        "base-nsga2" if component == "BASE" else
        "transformer-ppo-i+c+m-frozen" if component == "ICM" else
        trainer.name)
    policy_training = (
        False if component == "BASE" else
        any(item.training for item in trainer.values()) if component == "ICM" else
        bool(trainer.training))
    policy_checkpoint = (
        None if component == "BASE" else
        {name: item.checkpoint_provenance for name, item in trainer.items()}
        if component == "ICM" else trainer.checkpoint_provenance)
    config = {
        "schema_version": 3, "method": method,
        "protocol": "nsga2-learning-components-v1",
        "component": component,
        "evaluation_mode": evaluation_mode,
        "instance_id": instance["instance_id"], "configuration": instance["configuration"],
        "instance_digest": instance["instance_digest"], "K": instance["K"],
        "deadline_window_alpha": instance["deadline_window_alpha"],
        "population": pop_size, "generations": generations,
        "alpha": 0.90,
        "evaluation_budget": evaluation_budget,
        "training_size": scenarios.size,
        "scenario_digest": scenario_digest(scenarios), "seeds": seeds,
        "operator_probabilities": dict(zip(base.OPS, base._FIXED_OP_PROBS)),
        "crossover_rate": base.CROSSOVER_RATE, "mutation_rate": base.MUTATION_RATE,
        "external_archive_role": "reward-only; excluded from environmental selection",
        "reward_definition": "scaled positive joint archive HV increment only",
        "reward_clipping": False,
        "intermediate_post_crossover_evaluation": False,
        "policy_training": policy_training,
        "policy_checkpoint": policy_checkpoint,
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
    elif component == "ICM":
        controller = CombinedPPOController(
            out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
            trainers=trainer, normalizer=episode_normalizer,
            reward_config=reward_config or RewardConfig(), batches=batches,
            generations=generations, pop_size=pop_size,
            graph_observation=graph_observation, method=method,
            instance=instance, seeds=seeds)
    else:
        controller_type = controller_types[component]
        controller = controller_type(
            out, instance["instance_id"], config["scenario_digest"], evaluation_budget,
            trainer=trainer, archive=archive,
            reward_config=reward_config or RewardConfig(), batches=batches,
            generations=generations, pop_size=pop_size, method=method,
            instance=instance, seeds=seeds, graph_observation=graph_observation)
    base.ACTIVE_MUTATION_LOGGER = controller
    try:
        population = base.run_nsga2(
            nodes, regions, hold, proc, trans_cost, arcs, timetables, batches,
            wait_cost, wait_emission, carbon, emission_factors, speeds, trans_map,
            border_delay, theta, paths, options, pop_size=pop_size,
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
        + controller.feature_construction_seconds
        + controller.inference_seconds
        + controller.ppo_update_seconds)
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
    elif component == "ICM":
        summary["component_attempts"] = {
            name: item.total_attempts for name, item in controller.controllers.items()}
        summary["effective_component_modifications"] = {
            name: item.total_effective for name, item in controller.controllers.items()}
        summary["no_eligible_candidate"] = {
            name: item.no_candidate for name, item in controller.controllers.items()}
    write_json(out / "COMPLETE.json", summary)
    return summary
