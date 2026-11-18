#!/usr/bin/env python3
"""Pilot: direct-input EV and CCP30/CCP50 optimisation, then common OOS validation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np

import baseline_uncertainty as base
from run_ccp_candidate_pool import (
    decision_representation, decision_signature, nondominated_indices,
)


METHODS = ("EV", "CCP30", "CCP50")


def scenario_prefix(master: base.ScenarioSet, size: int) -> base.ScenarioSet:
    """Exact prefix view of a master training set; draws no randomness."""
    if not 1 <= size <= master.size:
        raise ValueError(f"prefix size {size} outside 1..{master.size}")
    return base.ScenarioSet(
        size=size, seed=master.seed,
        travel_multiplier={k: np.asarray(v[:size], dtype=float).copy()
                           for k, v in master.travel_multiplier.items()},
        border_delay_h={k: np.asarray(v[:size], dtype=float).copy()
                        for k, v in master.border_delay_h.items()},
        arc_border_event=dict(master.arc_border_event),
        border_event_mean_h=dict(master.border_event_mean_h),
        stochastic=master.stochastic,
    )


def scenario_digest(scenarios: base.ScenarioSet) -> str:
    digest = hashlib.sha256()
    digest.update(f"{scenarios.size}|{scenarios.seed}|{scenarios.stochastic}".encode())
    for collection in (scenarios.travel_multiplier, scenarios.border_delay_h):
        for key in sorted(collection):
            digest.update(json.dumps(key, separators=(",", ":")).encode())
            digest.update(np.asarray(collection[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def candidate_rows(method: str, source_run: str, individuals, config: dict[str, Any]):
    """Deduplicate only exact decisions, retaining equal-objective decisions."""
    rows = []
    seen = set()
    for source_index, ind in enumerate(individuals):
        signature = decision_signature(ind)
        if signature in seen:
            continue
        seen.add(signature)
        rows.append({
            "method": method,
            "source_run": source_run,
            "source_solution_id": f"{source_run}-solution-{source_index:04d}",
            "decision_fingerprint": signature,
            "decision": decision_representation(ind),
            "optimisation_objectives": {
                "cost": float(ind.objectives[0]),
                "emission": float(ind.objectives[1]),
                "makespan": float(ind.objectives[2]),
            },
            "training_feasible": bool(ind.feasible),
            "training_punctuality_diagnostic_min_on_time_probability": (
                float(min(ind.batch_on_time_prob.values()))
                if ind.batch_on_time_prob else None),
            "training_punctuality_diagnostic_batch_on_time_probability": {
                str(k): float(v) for k, v in ind.batch_on_time_prob.items()},
            "training_violation": dict(ind.vio_breakdown),
            "random_seeds": dict(config["seeds"]),
            "configuration": {
                "population": config["population"],
                "generations": config["generations"],
                "alpha": config["alpha"],
            },
            "individual": deepcopy(ind),
        })
    return rows


def json_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "individual"}


def summarise(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)), "minimum": float(np.min(values)),
        "maximum": float(np.max(values)), "median": float(np.median(values)),
        "q90": base.empirical_ccp_quantile(values, .9),
    }


def generalisation_delta(validation: float, training: float) -> dict[str, float | None]:
    return {
        "absolute": float(validation - training),
        "percentage": (float(100.0 * (validation - training) / training)
                       if abs(training) > 1e-15 else None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_expanded.xlsx")
    parser.add_argument("--out", default="pilot_ev_ccp_oos")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=.90)
    parser.add_argument("--training-seed", type=int, default=310050)
    parser.add_argument("--validation-seed", type=int, default=950005)
    parser.add_argument("--algorithm-seed", type=int, default=410001,
                        help="One paired NSGA-II seed used by all methods.")
    parser.add_argument("--validation-scenarios", type=int, default=5000)
    parser.add_argument("--expected-batches", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.training_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if args.pop < 2 or args.gens < 1 or args.validation_scenarios < 1:
        raise ValueError("pop>=2, gens>=1 and validation-scenarios>=1 required")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pilot directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        (out / method.lower()).mkdir()
    (out / "validation").mkdir()

    config = {
        "label": "one paired pilot; not a formal replication series",
        "data_file": str(Path(args.data).resolve()),
        "expected_batches": args.expected_batches,
        "population": args.pop, "generations": args.gens, "alpha": args.alpha,
        "training_sizes": [30, 50], "validation_size": args.validation_scenarios,
        "candidate_pool_limit": None,
        "seeds": {
            "training_master_S50": args.training_seed,
            "validation": args.validation_seed,
            "algorithm_shared_EV_CCP30_CCP50": args.algorithm_seed,
            "EV_optimisation": args.algorithm_seed,
            "CCP30_optimisation": args.algorithm_seed,
            "CCP50_optimisation": args.algorithm_seed,
            "path_library": 0,
        },
        "operators": {"crossover_rate": base.CROSSOVER_RATE,
                      "mutation_rate": base.MUTATION_RATE},
        "uncertainty_parameters_unchanged": {
            "mode_time_cv": dict(base.MODE_TIME_CV),
            "mode_time_max_factor": dict(base.MODE_TIME_MAX_FACTOR),
            "border_delay_cv": dict(base.BORDER_DELAY_CV),
            "border_delay_max_factor": base.BORDER_DELAY_MAX_FACTOR,
            "wait_emission_g_per_teu_h": None,
        },
        "ev_definition": "S=1 direct model expectations: travel multipliers=1.0 and border delays=model means; no sampling",
        "ccp_definition": "all training scenarios evaluated during optimisation; empirical ceil(alpha*S)-th objective order statistics",
        "punctuality_semantics": "diagnostic only; not CCP feasibility",
        "paired_design": "same algorithm seed, path library, and reliable-option ordering for all methods",
    }

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(args.data)
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost, wait_emission, carbon_tax,
     emission_factors, mode_speeds, trans_map, border_delay_map, theta_rm,
     _mode_speed_source) = network
    if len(batches) != args.expected_batches:
        raise ValueError(f"expected {args.expected_batches} batches, got {len(batches)}")
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H
    tt_dict = base.build_timetable_dict(timetables)
    arc_lookup = base.build_arc_lookup(arcs)
    random.seed(0); np.random.seed(0)
    path_lib = base.build_path_library(
        node_names, node_region, arcs, batches, tt_dict, arc_lookup)
    base.sanity_check_path_lib(batches, path_lib)

    ev_set = base.build_expected_value_scenario_set(
        arcs, border_delay_map, seed=0,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    training50 = base.build_scenario_set(
        arcs, border_delay_map, 50, args.training_seed, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    training30 = scenario_prefix(training50, 30)
    validation = base.build_scenario_set(
        arcs, border_delay_map, args.validation_scenarios,
        args.validation_seed, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)
    config["scenario_digests"] = {
        "training_S30_prefix": scenario_digest(training30),
        "training_S50_master": scenario_digest(training50),
        "validation": scenario_digest(validation),
    }
    config["independence_evidence"] = {
        "different_seeds": args.training_seed != args.validation_seed,
        "separate_generation_calls": True,
        "validation_digest_differs_from_training": (
            scenario_digest(validation) not in
            {scenario_digest(training30), scenario_digest(training50)}),
        "S30_is_exact_S50_prefix": (
            all(np.array_equal(training30.travel_multiplier[k],
                               training50.travel_multiplier[k][:30])
                for k in training30.travel_multiplier)
            and all(np.array_equal(training30.border_delay_h[k],
                                   training50.border_delay_h[k][:30])
                    for k in training30.border_delay_h)),
    }
    config["paired_seed_evidence"] = {
        "all_method_algorithm_seeds_equal": len({
            config["seeds"]["EV_optimisation"],
            config["seeds"]["CCP30_optimisation"],
            config["seeds"]["CCP50_optimisation"],
        }) == 1,
        "shared_path_library": True,
        "shared_reliable_option_ordering": True,
    }
    config["uncertainty_parameters_unchanged"][
        "wait_emission_g_per_teu_h"] = float(wait_emission)
    (out / "configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    eval_kwargs = dict(
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        carbon_tax_map=carbon_tax, trans_map=trans_map,
        border_delay_map=border_delay_map, theta_rm=theta_rm,
        node_trans_cost=node_trans_cost)
    method_specs = {
        "EV": (ev_set, "ev"),
        "CCP30": (training30, "ccp"),
        "CCP50": (training50, "ccp"),
    }
    shared_reliable = base.build_reliable_path_options(
        batches, path_lib, tt_dict, trans_map, border_delay_map,
        ev_set, mode="ev")
    candidates = []
    runtimes = {}
    for method, (scenarios, risk_metric) in method_specs.items():
        started = time.perf_counter()
        base.ACTIVE_SCENARIO_SET = scenarios
        base._PATH_SCENARIO_CACHE = {}
        base.RISK_METRIC = risk_metric
        base.CONFIDENCE_COST = base.CONFIDENCE_EMISSION = base.CONFIDENCE_TIME = args.alpha
        random.seed(args.algorithm_seed); np.random.seed(args.algorithm_seed)
        population = base.run_nsga2(
            node_names, node_region, node_hold_cost, node_proc_cost,
            node_trans_cost, arcs, timetables, batches, waiting_cost,
            wait_emission, carbon_tax, emission_factors, mode_speeds,
            trans_map, border_delay_map, theta_rm, path_lib, shared_reliable,
            pop_size=args.pop, generations=args.gens)[0]
        fronts = base.fast_non_dominated_sort(population)
        front = [deepcopy(population[i]) for i in fronts[0]
                 if population[i].feasible]
        runtimes[f"{method}_optimisation_seconds"] = time.perf_counter() - started
        rows = candidate_rows(method, f"pilot-{method}-run-1", front, config)
        candidates.extend(rows)
        (out / method.lower() / "final_feasible_nondominated.json").write_text(
            json.dumps([json_candidate(r) for r in rows], indent=2), encoding="utf-8")

    (out / "candidate_provenance.json").write_text(
        json.dumps([json_candidate(r) for r in candidates], indent=2), encoding="utf-8")
    if not candidates:
        raise RuntimeError("no feasible nondominated candidate was produced")

    # Stage C: one direct evaluation of each frozen decision over the same set.
    stage_c_started = time.perf_counter()
    base.ACTIVE_SCENARIO_SET = validation
    base._PATH_SCENARIO_CACHE = {}
    base.RISK_METRIC = "ev"  # reporting reduction only; arrays are retained.
    summaries = []
    scenario_csv = out / "validation" / "scenario_level_results.csv"
    with scenario_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "source_solution_id", "decision_fingerprint",
            "scenario_id", "cost", "emission", "makespan"])
        writer.writeheader()
        for row in candidates:
            fixed = deepcopy(row["individual"])
            before = decision_signature(fixed)
            base.evaluate_individual(
                fixed, batches, arcs, tt_dict, waiting_cost, wait_emission,
                **eval_kwargs)
            after = decision_signature(fixed)
            if before != after or before != row["decision_fingerprint"]:
                raise RuntimeError("Stage C changed a fixed candidate decision")
            for scenario_id, (cost, emission, makespan) in enumerate(zip(
                    fixed.cost_s, fixed.emission_s, fixed.makespan_s)):
                writer.writerow({
                    "method": row["method"],
                    "source_solution_id": row["source_solution_id"],
                    "decision_fingerprint": before, "scenario_id": scenario_id,
                    "cost": float(cost), "emission": float(emission),
                    "makespan": float(makespan)})
            min_probability = (float(min(fixed.batch_on_time_prob.values()))
                               if fixed.batch_on_time_prob else 0.0)
            training = row["optimisation_objectives"]
            cost_summary = summarise(fixed.cost_s)
            emission_summary = summarise(fixed.emission_s)
            makespan_summary = summarise(fixed.makespan_s)
            validation_key = "mean" if row["method"] == "EV" else "q90"
            summaries.append({
                "method": row["method"],
                "source_solution_id": row["source_solution_id"],
                "decision_fingerprint": before,
                "stage_c_signature_before": before,
                "stage_c_signature_after": after,
                "stage_c_decision_unchanged": before == after,
                "training_objectives": training,
                "training_semantics": ("direct_expected_inputs" if row["method"] == "EV"
                                       else "empirical_q90_objectives"),
                "cost": cost_summary,
                "emission": emission_summary,
                "makespan": makespan_summary,
                "primary_generalisation_comparison": (
                    "training_expected_input_vs_oos_mean" if row["method"] == "EV"
                    else "training_q90_vs_oos_q90"),
                "generalisation_delta": {
                    objective: generalisation_delta(summary[validation_key],
                                                    training[objective])
                    for objective, summary in (
                        ("cost", cost_summary), ("emission", emission_summary),
                        ("makespan", makespan_summary))},
                "emission_constant": bool(np.ptp(fixed.emission_s)
                    <= 1e-12 * max(1.0, float(np.max(np.abs(fixed.emission_s))))),
                "punctuality_diagnostic_min_batch_on_time_probability": min_probability,
                "punctuality_diagnostic_batch_on_time_probability": {
                    str(k): float(v) for k, v in fixed.batch_on_time_prob.items()},
                "punctuality_diagnostic_all_batches_at_least_alpha": (
                    min_probability >= args.alpha),
            })
    runtimes["stage_c_validation_seconds"] = time.perf_counter() - stage_c_started

    mean_fronts = nondominated_indices([
        (s["cost"]["mean"], s["emission"]["mean"], s["makespan"]["mean"])
        for s in summaries])
    q90_fronts = nondominated_indices([
        (s["cost"]["q90"], s["emission"]["q90"], s["makespan"]["q90"])
        for s in summaries])
    for rank, indices in enumerate(mean_fronts):
        for index in indices: summaries[index]["oos_mean_pareto_rank"] = rank
    for rank, indices in enumerate(q90_fronts):
        for index in indices: summaries[index]["oos_p90_pareto_rank"] = rank
    (out / "validation" / "per_candidate_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    mean_pareto = [s for s in summaries if s["oos_mean_pareto_rank"] == 0]
    q90_pareto = [s for s in summaries if s["oos_p90_pareto_rank"] == 0]
    (out / "validation" / "oos_mean_pareto.json").write_text(
        json.dumps(mean_pareto, indent=2),
        encoding="utf-8")
    (out / "validation" / "oos_q90_pareto.json").write_text(
        json.dumps(q90_pareto, indent=2),
        encoding="utf-8")
    (out / "runtime_summary.json").write_text(
        json.dumps(runtimes, indent=2), encoding="utf-8")
    counts = {method: sum(r["method"] == method for r in candidates)
              for method in METHODS}
    method_fingerprints = {
        method: frozenset(r["decision_fingerprint"] for r in candidates
                          if r["method"] == method)
        for method in METHODS}
    shared_fingerprints = {
        f"{left}_{right}": sorted(method_fingerprints[left]
                                  & method_fingerprints[right])
        for index, left in enumerate(METHODS)
        for right in METHODS[index + 1:]
    }
    report = {
        "candidate_counts": counts,
        "unique_fingerprints_by_method": {
            method: len(method_fingerprints[method]) for method in METHODS},
        "unique_fingerprints_global": len({r["decision_fingerprint"] for r in candidates}),
        "shared_fingerprints_across_methods": shared_fingerprints,
        "oos_front_representation": {
            "mean": {method: sum(s["method"] == method for s in mean_pareto)
                     for method in METHODS},
            "q90": {method: sum(s["method"] == method for s in q90_pareto)
                    for method in METHODS},
        },
        "punctuality_diagnostic_counts_all_batches_at_least_alpha": {
            method: sum(
                s["method"] == method
                and s["punctuality_diagnostic_all_batches_at_least_alpha"]
                for s in summaries) for method in METHODS},
        "methods_have_different_decisions": (
            len(set(method_fingerprints.values())) > 1),
        "all_fixed_candidate_emissions_scenario_invariant": all(
            s["emission_constant"] for s in summaries),
        "total_candidate_scenario_evaluations": (
            len(candidates) * args.validation_scenarios),
        "previous_S200_runtime_reference": {
            "training_scenarios": 200, "population": 300,
            "generations": 500, "runtime_seconds": 650.3,
            "comparison_is_descriptive_only": True,
        },
        "runtime_seconds": runtimes,
    }
    (out / "pilot_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# EV/CCP out-of-sample workflow pilot\n\n"
        f"This is one paired pilot, not a formal replication series. It used "
        f"population {args.pop} and {args.gens} generations once per method. EV optimises "
        "one environment constructed from the model's expected uncertainty inputs. "
        "CCP30 and CCP50 optimise empirical 90th-percentile objectives over nested "
        f"training samples with shared algorithm seed {args.algorithm_seed}. All "
        "exported decisions are then frozen and evaluated over "
        "one independently generated common S_val=5000 set. Stage C calls only the "
        "scenario simulator/evaluator; it creates no population and invokes no search "
        "operator. Signature guards fail if any route, mode, or share changes.\n\n"
        "Emissions are constant when waiting emissions are zero because the existing "
        "formula depends only on fixed path distance, mode emission factor, and flow.\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
