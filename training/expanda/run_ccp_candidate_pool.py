#!/usr/bin/env python3
"""Two-stage scenario-Pareto candidate-pool CCP experiment driver.

Stage A optimises independently under each member of one frozen scenario set.
Stages B--E deduplicate decisions, re-evaluate each frozen decision under every
scenario, reduce the marginal objective arrays to EV/CCP values, and rank the
same pool under both reductions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path as FSPath
import random
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import baseline_uncertainty as base


Objective = Tuple[float, float, float]
SIGNATURE_SHARE_DECIMALS = 12


def scenario_slice(scenarios: base.ScenarioSet, scenario_id: int) -> base.ScenarioSet:
    """Return an exact one-scenario view without drawing new randomness."""
    if not 0 <= scenario_id < scenarios.size:
        raise IndexError(scenario_id)
    return base.ScenarioSet(
        size=1,
        seed=scenarios.seed,
        travel_multiplier={
            key: np.asarray([values[scenario_id]], dtype=float)
            for key, values in scenarios.travel_multiplier.items()
        },
        border_delay_h={
            key: np.asarray([values[scenario_id]], dtype=float)
            for key, values in scenarios.border_delay_h.items()
        },
        arc_border_event=dict(scenarios.arc_border_event),
        border_event_mean_h=dict(scenarios.border_event_mean_h),
        stochastic=scenarios.stochastic,
    )


def decision_signature(ind: base.Individual) -> str:
    """Canonical fingerprint of decision variables, never objective values.

    Allocation blocks and paths are sorted, so representational ordering is
    irrelevant. Shares are rounded to 12 decimal places before hashing: this
    removes binary floating-point noise far below the encoding's meaningful
    precision while preserving genuine allocation differences.
    """
    blocks = []
    for key in sorted(ind.od_allocations):
        allocations = []
        for allocation in ind.od_allocations[key]:
            share = round(float(allocation.share), SIGNATURE_SHARE_DECIMALS)
            if share == 0.0:  # canonicalise negative zero as well
                share = 0.0
            allocations.append((
                tuple(allocation.path.nodes),
                tuple(allocation.path.modes),
                f"{share:.{SIGNATURE_SHARE_DECIMALS}f}",
            ))
        blocks.append((tuple(key), tuple(sorted(allocations))))
    canonical = json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))
    import hashlib
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decision_representation(ind: base.Individual) -> List[Dict[str, Any]]:
    result = []
    for key in sorted(ind.od_allocations):
        result.append({
            "origin": key[0], "destination": key[1], "batch_id": int(key[2]),
            "allocations": [{
                "share": float(a.share),
                "path_id": int(a.path.path_id),
                "nodes": list(a.path.nodes),
                "modes": list(a.path.modes),
            } for a in ind.od_allocations[key]],
        })
    return result


def nondominated_indices(values: Sequence[Objective]) -> List[List[int]]:
    """Return all Pareto ranks for minimisation objective triples."""
    remaining = set(range(len(values)))
    fronts: List[List[int]] = []
    while remaining:
        front = []
        for i in sorted(remaining):
            dominated = any(
                all(values[j][m] <= values[i][m] for m in range(3))
                and any(values[j][m] < values[i][m] for m in range(3))
                for j in remaining if j != i
            )
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining.difference_update(front)
    return fronts


def assign_ranks(records: List["CandidateRecord"], attr: str, rank_attr: str) -> None:
    values = [getattr(record, attr) for record in records]
    for rank, front in enumerate(nondominated_indices(values)):
        for index in front:
            setattr(records[index], rank_attr, rank)


@dataclass
class CandidateRecord:
    signature: str
    individual: base.Individual
    scenario_ids: set[int] = field(default_factory=set)
    occurrence_count: int = 0
    cost_s: List[float] = field(default_factory=list)
    emission_s: List[float] = field(default_factory=list)
    makespan_s: List[float] = field(default_factory=list)
    ev: Objective = (math.inf, math.inf, math.inf)
    ccp: Objective = (math.inf, math.inf, math.inf)
    ev_rank: int = -1
    ccp_rank: int = -1
    evaluation_count: int = 0
    stage_c_signature_before: str = ""
    stage_c_signature_after: str = ""


def pool_scenario_paretos(
    scenario_paretos: Sequence[Tuple[int, Sequence[base.Individual]]]
) -> Tuple[List[CandidateRecord], int]:
    by_signature: Dict[str, CandidateRecord] = {}
    occurrences = 0
    for scenario_id, pareto in scenario_paretos:
        for ind in pareto:
            occurrences += 1
            signature = decision_signature(ind)
            record = by_signature.get(signature)
            if record is None:
                record = CandidateRecord(signature, deepcopy(ind))
                by_signature[signature] = record
            record.scenario_ids.add(int(scenario_id))
            record.occurrence_count += 1
    return list(by_signature.values()), occurrences


def reduce_candidate(record: CandidateRecord, alpha: float) -> None:
    record.ev = tuple(float(np.mean(values)) for values in (
        record.cost_s, record.emission_s, record.makespan_s))
    record.ccp = tuple(base.empirical_ccp_quantile(np.asarray(values), alpha)
                       for values in (
                           record.cost_s, record.emission_s, record.makespan_s))


def re_evaluate_pool(
    records: List[CandidateRecord], scenarios: base.ScenarioSet,
    evaluate_one: Callable[[base.Individual, base.ScenarioSet], Objective],
    alpha: float,
) -> None:
    """Evaluate frozen decisions only; no repair, crossover, or mutation."""
    for record in records:
        signature_before = decision_signature(record.individual)
        record.stage_c_signature_before = signature_before
        for scenario_id in range(scenarios.size):
            evaluation_copy = deepcopy(record.individual)
            copy_signature_before = decision_signature(evaluation_copy)
            realised = evaluate_one(
                evaluation_copy, scenario_slice(scenarios, scenario_id))
            copy_signature_after = decision_signature(evaluation_copy)
            if copy_signature_after != copy_signature_before:
                raise RuntimeError(
                    "Candidate evaluation copy changed during Stage C "
                    f"(scenario_id={scenario_id})")
            record.cost_s.append(float(realised[0]))
            record.emission_s.append(float(realised[1]))
            record.makespan_s.append(float(realised[2]))
            record.evaluation_count += 1
        record.stage_c_signature_after = decision_signature(record.individual)
        if record.stage_c_signature_after != signature_before:
            raise RuntimeError("Candidate decision changed during re-evaluation")
        reduce_candidate(record, alpha)
    assign_ranks(records, "ccp", "ccp_rank")
    assign_ranks(records, "ev", "ev_rank")


def candidate_json(record: CandidateRecord) -> Dict[str, Any]:
    return {
        "decision_signature": record.signature,
        "scenario_pareto_appearances": len(record.scenario_ids),
        "scenario_ids": sorted(record.scenario_ids),
        "pareto_occurrence_count": record.occurrence_count,
        "evaluation_count": record.evaluation_count,
        "stage_c_signature_before": record.stage_c_signature_before,
        "stage_c_signature_after": record.stage_c_signature_after,
        "stage_c_decision_unchanged": (
            record.stage_c_signature_before == record.stage_c_signature_after),
        "decision": decision_representation(record.individual),
        "cost_s": record.cost_s,
        "emission_s": record.emission_s,
        "makespan_s": record.makespan_s,
        "EV_Cost": record.ev[0], "EV_Emission": record.ev[1],
        "EV_Time": record.ev[2],
        "CCP90_Cost": record.ccp[0], "CCP90_Emission": record.ccp[1],
        "CCP90_Time": record.ccp[2],
        "EV_Pareto_Rank": record.ev_rank,
        "CCP90_Pareto_Rank": record.ccp_rank,
    }


def run_candidate_pool_workflow(
    scenarios: base.ScenarioSet,
    optimise_one: Callable[[int, base.ScenarioSet], Sequence[base.Individual]],
    evaluate_one: Callable[[base.Individual, base.ScenarioSet], Objective],
    alpha: float = 0.90,
) -> Tuple[List[CandidateRecord], int, List[Tuple[int, Sequence[base.Individual]]]]:
    scenario_paretos = []
    for scenario_id in range(scenarios.size):
        print(f"[STAGE A] scenario {scenario_id + 1}/{scenarios.size} "
              "fitness=realised_single_scenario(Cost,Emission,Time); "
              "EV/CCP reduction disabled")
        singleton = scenario_slice(scenarios, scenario_id)
        if singleton.size != 1:
            raise AssertionError("Stage A requires exactly one frozen scenario")
        pareto = list(optimise_one(scenario_id, singleton))
        if not pareto:
            raise RuntimeError(f"Scenario {scenario_id} produced no feasible Pareto set")
        scenario_paretos.append((scenario_id, pareto))
        print(f"[STAGE A] scenario_id={scenario_id} pareto={len(pareto)}")
    records, occurrences = pool_scenario_paretos(scenario_paretos)
    print(f"[STAGE B] occurrences={occurrences} unique_decisions={len(records)}")
    re_evaluate_pool(records, scenarios, evaluate_one, alpha)
    print(f"[STAGES C-E] evaluations={sum(r.evaluation_count for r in records)} "
          f"ccp_front={sum(r.ccp_rank == 0 for r in records)} "
          f"ev_front={sum(r.ev_rank == 0 for r in records)}")
    return records, occurrences, scenario_paretos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/data_expanded.xlsx")
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--mc-seed", type=int, default=1000003)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--pop", type=int, default=8)
    parser.add_argument("--gens", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.90,
                        help="Marginal objective quantile used only after pooling.")
    parser.add_argument("--out", default="outputs/ccp_candidate_pool_smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenarios < 1 or args.pop < 2 or args.gens < 1:
        raise ValueError("scenarios>=1, pop>=2 and gens>=1 are required")
    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(args.data)
    (node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, batches, waiting_cost, wait_emission, carbon_tax,
     emission_factors, mode_speeds, trans_map, border_delay_map, theta_rm,
    _mode_speed_source) = network
    for batch in batches:
        batch.penalty_per_teu_h = base.DEFAULT_LATE_PENALTY_USD_PER_TEU_H

    tt_dict = base.build_timetable_dict(timetables)
    arc_lookup = base.build_arc_lookup(arcs)
    random.seed(0); np.random.seed(0)
    path_lib = base.build_path_library(
        node_names, node_region, arcs, batches, tt_dict, arc_lookup)
    base.sanity_check_path_lib(batches, path_lib)
    frozen = base.build_scenario_set(
        arcs, border_delay_map, args.scenarios, args.mc_seed, stochastic=True,
        border_event_definitions=base.BORDER_EVENT_DEFINITIONS)

    eval_kwargs = dict(
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        carbon_tax_map=carbon_tax, trans_map=trans_map,
        border_delay_map=border_delay_map, theta_rm=theta_rm,
        node_trans_cost=node_trans_cost)

    def evaluate_one(ind: base.Individual, singleton: base.ScenarioSet) -> Objective:
        base.ACTIVE_SCENARIO_SET = singleton
        base._PATH_SCENARIO_CACHE = {}
        previous_mode = base.RISK_METRIC
        base.RISK_METRIC = "deterministic"
        try:
            base.evaluate_individual(ind, batches, arcs, tt_dict,
                                     waiting_cost, wait_emission, **eval_kwargs)
            return ind.objectives
        finally:
            base.RISK_METRIC = previous_mode

    scenario_exports: Dict[int, List[Dict[str, Any]]] = {}

    def optimise_one(scenario_id: int, singleton: base.ScenarioSet):
        # RISK_METRIC="deterministic" on an exact S=1 frozen slice makes every
        # NSGA-II comparison use only (Cost(x,w_s), Emission(x,w_s), Time(x,w_s)).
        # EV means and CCP quantiles are computed later, after global pooling.
        if singleton.size != 1:
            raise AssertionError("Stage A received more than one scenario")
        base.ACTIVE_SCENARIO_SET = singleton
        base._PATH_SCENARIO_CACHE = {}
        previous_mode = base.RISK_METRIC
        base.RISK_METRIC = "deterministic"
        random.seed(args.seed + scenario_id); np.random.seed(args.seed + scenario_id)
        try:
            options = base.build_reliable_path_options(
                batches, path_lib, tt_dict, trans_map, border_delay_map,
                singleton, mode="deterministic")
            population = base.run_nsga2(
                node_names, node_region, node_hold_cost, node_proc_cost,
                node_trans_cost, arcs, timetables, batches, waiting_cost,
                wait_emission, carbon_tax, emission_factors, mode_speeds,
                trans_map, border_delay_map, theta_rm, path_lib, options,
                pop_size=args.pop, generations=args.gens)[0]
            fronts = base.fast_non_dominated_sort(population)
            pareto = [deepcopy(population[i]) for i in fronts[0]
                      if population[i].feasible]
            scenario_exports[scenario_id] = [{
                "scenario_id": scenario_id,
                "decision_signature": decision_signature(ind),
                "decision": decision_representation(ind),
                "Cost": float(ind.objectives[0]),
                "Emission": float(ind.objectives[1]),
                "Time": float(ind.objectives[2]),
            } for ind in pareto]
            return pareto
        finally:
            base.RISK_METRIC = previous_mode

    records, occurrences, _ = run_candidate_pool_workflow(
        frozen, optimise_one, evaluate_one, alpha=args.alpha)
    payload = [candidate_json(record) for record in records]
    (out / "scenario_pareto_sets.json").write_text(
        json.dumps(scenario_exports, indent=2), encoding="utf-8")
    (out / "all_unique_candidates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    (out / "final_ccp_pareto_front.json").write_text(
        json.dumps([row for row in payload if row["CCP90_Pareto_Rank"] == 0], indent=2),
        encoding="utf-8")
    (out / "final_ev_pareto_front.json").write_text(
        json.dumps([row for row in payload if row["EV_Pareto_Rank"] == 0], indent=2),
        encoding="utf-8")
    summary = {
        "scenario_count": frozen.size, "alpha": args.alpha,
        "pareto_occurrences_before_deduplication": occurrences,
        "unique_candidate_decisions": len(records),
        "evaluations_per_unique_candidate": frozen.size,
        "candidate_pool_limit": None,
    }
    (out / "candidate_pool_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
