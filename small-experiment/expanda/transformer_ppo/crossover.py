"""Common-node crossover planning with explicit legal cut candidates."""
from __future__ import annotations

from dataclasses import dataclass
import random

import baseline_uncertainty as base


@dataclass(frozen=True)
class CutDecision:
    batch: object
    key: tuple
    allocation_index_1: int
    allocation_index_2: int
    candidates: tuple[dict, ...]
    shadow_base_index: int


def _stable_common_nodes(path1, path2):
    """Return legal nodes in chromosome order, not set iteration order."""
    available = set(path2.nodes[1:-1])
    seen = set()
    result = []
    for node in path1.nodes[1:-1]:
        if node in available and node not in seen:
            seen.add(node)
            result.append(node)
    return result


def _path_summary(arcs, tt_dict=None):
    modes = [str(arc.mode).lower() for arc in arcs]
    denominator = max(1, len(arcs))
    return {
        "cost": sum(float(arc.cost_per_teu_km) * float(arc.distance) for arc in arcs),
        "emission": sum(
            float(arc.emission_per_teu_km) * float(arc.distance) for arc in arcs),
        "time": sum(base.nominal_arc_travel_time(arc, tt_dict or {})
                    for arc in arcs),
        "distance": sum(float(arc.distance) for arc in arcs),
        "capacity": min((float(arc.capacity) for arc in arcs), default=0.0),
        "road": sum(mode == "road" for mode in modes) / denominator,
        "rail": sum(mode == "rail" for mode in modes) / denominator,
        "water": sum(mode == "water" for mode in modes) / denominator,
        "border": sum(bool(arc.is_border_arc) for arc in arcs) / denominator,
    }


def prepare_common_node_crossover(parent1, parent2, batches, rng=None,
                                  tt_dict=None):
    """Clone both parents and expose the exact cuts Base would choose between.

    Base first chooses one allocation from each parent independently for every
    Batch.  Learning changes only the subsequent common-node choice.
    """
    rng = rng or random
    child1, child2 = base.Individual(), base.Individual()
    decisions = []
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        genes1 = parent1.od_allocations.get(key, [])
        genes2 = parent2.od_allocations.get(key, [])
        if not genes1 and not genes2:
            continue
        if not genes1:
            child1.od_allocations[key] = [base.clone_gene(item) for item in genes2]
            child2.od_allocations[key] = [base.clone_gene(item) for item in genes2]
            continue
        if not genes2:
            child1.od_allocations[key] = [base.clone_gene(item) for item in genes1]
            child2.od_allocations[key] = [base.clone_gene(item) for item in genes1]
            continue

        child1.od_allocations[key] = [base.clone_gene(item) for item in genes1]
        child2.od_allocations[key] = [base.clone_gene(item) for item in genes2]
        index1, index2 = rng.randrange(len(genes1)), rng.randrange(len(genes2))
        path1, path2 = genes1[index1].path, genes2[index2].path
        candidates = []
        for node in _stable_common_nodes(path1, path2):
            position1 = path1.nodes.index(node)
            position2 = path2.nodes.index(node)
            child1_arcs = list(path1.arcs[:position1]) + list(path2.arcs[position2:])
            child2_arcs = list(path2.arcs[:position2]) + list(path1.arcs[position1:])
            parent1_summary, parent2_summary = (
                _path_summary(path1.arcs, tt_dict),
                _path_summary(path2.arcs, tt_dict))
            child1_summary, child2_summary = (
                _path_summary(child1_arcs, tt_dict),
                _path_summary(child2_arcs, tt_dict))
            share1, share2 = float(genes1[index1].share), float(genes2[index2].share)
            quantity = float(batch.quantity)
            parent_summary = {
                "cost": quantity * (share1 * parent1_summary["cost"]
                                    + share2 * parent2_summary["cost"]),
                "emission": quantity * (share1 * parent1_summary["emission"]
                                        + share2 * parent2_summary["emission"]),
                "time": max(parent1_summary["time"], parent2_summary["time"]),
            }
            child_summary = {
                "cost": quantity * (share1 * child1_summary["cost"]
                                    + share2 * child2_summary["cost"]),
                "emission": quantity * (share1 * child1_summary["emission"]
                                        + share2 * child2_summary["emission"]),
                "time": max(child1_summary["time"], child2_summary["time"]),
            }
            child1_nodes = [child1_arcs[0].from_node] + [
                arc.to_node for arc in child1_arcs] if child1_arcs else []
            child2_nodes = [child2_arcs[0].from_node] + [
                arc.to_node for arc in child2_arcs] if child2_arcs else []
            candidates.append({
                "eligible": True,
                "batch_id": batch.batch_id,
                "allocation_index_1": index1,
                "allocation_index_2": index2,
                "join_node": node,
                "position_1": position1,
                "position_2": position2,
                "path_length_1": len(path1.arcs),
                "path_length_2": len(path2.arcs),
                "share_1": float(genes1[index1].share),
                "share_2": float(genes2[index2].share),
                "origin": batch.origin,
                "destination": batch.destination,
                "batch_quantity": float(batch.quantity),
                "parent_combined_cost": parent_summary["cost"],
                "parent_combined_emission": parent_summary["emission"],
                "parent_combined_time": parent_summary["time"],
                "child_combined_cost": child_summary["cost"],
                "child_combined_emission": child_summary["emission"],
                "child_combined_time": child_summary["time"],
                "child_cost_delta": child_summary["cost"] - parent_summary["cost"],
                "child_emission_delta": (
                    child_summary["emission"] - parent_summary["emission"]),
                "child_time_delta": child_summary["time"] - parent_summary["time"],
                "child_1_cost_per_teu": child1_summary["cost"],
                "child_1_emission_per_teu": child1_summary["emission"],
                "child_1_nominal_time_h": child1_summary["time"],
                "child_2_cost_per_teu": child2_summary["cost"],
                "child_2_emission_per_teu": child2_summary["emission"],
                "child_2_nominal_time_h": child2_summary["time"],
                "child_1_distance": child1_summary["distance"],
                "child_2_distance": child2_summary["distance"],
                "child_1_bottleneck_capacity": child1_summary["capacity"],
                "child_2_bottleneck_capacity": child2_summary["capacity"],
                "child_1_road_fraction": child1_summary["road"],
                "child_1_rail_fraction": child1_summary["rail"],
                "child_1_water_fraction": child1_summary["water"],
                "child_2_road_fraction": child2_summary["road"],
                "child_2_rail_fraction": child2_summary["rail"],
                "child_2_water_fraction": child2_summary["water"],
                "child_border_fraction": 0.5 * (
                    child1_summary["border"] + child2_summary["border"]),
                "deadline_window_h": float(batch.LT) - float(batch.ET),
                "candidate_graph_paths": [
                    [child1_nodes, [arc.mode for arc in child1_arcs]],
                    [child2_nodes, [arc.mode for arc in child2_arcs]],
                ],
            })
        if candidates:
            # Base chooses the common node immediately after choosing the two
            # allocations for this Batch.  Consume that draw here so multiple
            # batches preserve Base's exact interleaved RNG stream.
            shadow_base_index = rng.randrange(len(candidates))
            decisions.append(CutDecision(
                batch=batch, key=key,
                allocation_index_1=index1, allocation_index_2=index2,
                candidates=tuple(candidates),
                shadow_base_index=shadow_base_index))
    return child1, child2, decisions


def apply_cut(child1, child2, decision, candidate, tt_dict, arc_lookup):
    """Apply one chosen legal cut and preserve the parents' allocation shares."""
    allocations1 = child1.od_allocations[decision.key]
    allocations2 = child2.od_allocations[decision.key]
    index1, index2 = decision.allocation_index_1, decision.allocation_index_2
    path1, path2 = allocations1[index1].path, allocations2[index2].path
    node = candidate["join_node"]
    new1 = base.perform_single_point_crossover_paths(
        path1, path2, node, tt_dict, arc_lookup)
    new2 = base.perform_single_point_crossover_paths(
        path2, path1, node, tt_dict, arc_lookup)
    changed = False
    if new1 is not None:
        allocations1[index1] = base.PathAllocation(
            path=new1, share=allocations1[index1].share)
        changed = changed or new1 != path1
    if new2 is not None:
        allocations2[index2] = base.PathAllocation(
            path=new2, share=allocations2[index2].share)
        changed = changed or new2 != path2
    return changed


def finalize_children(child1, child2):
    """Match Base's post-cut merge/filter/normalization for every Batch."""
    for child in (child1, child2):
        for key, allocations in list(child.od_allocations.items()):
            child.od_allocations[key] = base.merge_and_normalize(allocations)
