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


def prepare_common_node_crossover(parent1, parent2, batches, rng=None):
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
            candidates.append({
                "eligible": True,
                "batch_id": batch.batch_id,
                "allocation_index_1": index1,
                "allocation_index_2": index2,
                "join_node": node,
                "position_1": path1.nodes.index(node),
                "position_2": path2.nodes.index(node),
                "path_length_1": len(path1.arcs),
                "path_length_2": len(path2.arcs),
                "share_1": float(genes1[index1].share),
                "share_2": float(genes2[index2].share),
                "origin": batch.origin,
                "destination": batch.destination,
                "batch_quantity": float(batch.quantity),
            })
        if candidates:
            decisions.append(CutDecision(
                batch=batch, key=key,
                allocation_index_1=index1, allocation_index_2=index2,
                candidates=tuple(candidates)))
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
