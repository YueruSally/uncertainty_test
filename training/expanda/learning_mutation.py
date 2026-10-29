"""Position-selection interface. V1 implements Random only; no ML claims."""
from dataclasses import dataclass
import math

import baseline_uncertainty as base


@dataclass(frozen=True)
class MutationTarget:
    batch_id: int
    allocation_index: int | None = None
    arc_index: int | None = None


def enumerate_targets(ind, batches, op, path_lib, tt_dict, arc_lookup):
    """Preserve hierarchical uniform batch/path/arc sampling, including failures.

    Do not silently condition on eligibility: doing so would change the Random
    control and erase failed attempts. Rule/Learning must reuse this support.
    """
    rows = []
    for batch in batches:
        allocs = ind.od_allocations.get((batch.origin, batch.destination, batch.batch_id), [])
        indices = range(len(allocs)) if op in ("del", "mod", "mode") and allocs else [None]
        for idx in indices:
            alloc = allocs[idx] if idx is not None else None
            arc_indices = range(len(alloc.path.arcs)) if op == "mode" and alloc and alloc.path.arcs else [None]
            for arc_idx in arc_indices:
                arc = alloc.path.arcs[arc_idx] if arc_idx is not None else None
                alternatives = [] if arc is None else [m for m in ("road", "rail", "water")
                    if m != arc.mode and (arc.from_node, arc.to_node, m) in arc_lookup
                    and (m == "road" or tt_dict.get((arc.from_node, arc.to_node, m)))]
                probability = 1.0 / len(batches) / len(indices) / len(arc_indices)
                rows.append({
                    "target": MutationTarget(batch.batch_id, idx, arc_idx),
                    "selection_probability": probability,
                    "batch_quantity": float(batch.quantity), "path_count": len(allocs),
                    "share": None if alloc is None else float(alloc.share),
                    "path_id": None if alloc is None else alloc.path.path_id,
                    "path_length": 0 if alloc is None else len(alloc.path.arcs),
                    "path_nodes": [] if alloc is None else list(alloc.path.nodes),
                    "path_cost_per_teu": None if alloc is None else alloc.path.base_cost_per_teu,
                    "path_emission_per_teu": None if alloc is None else alloc.path.base_emission_per_teu,
                    "path_nominal_time_h": None if alloc is None else alloc.path.base_travel_time_h,
                    "from_node": None if arc is None else arc.from_node,
                    "to_node": None if arc is None else arc.to_node,
                    "current_mode": None if arc is None else arc.mode,
                    "alternative_mode_count": len(alternatives),
                    "eligible": (len(allocs) > 1 if op == "del" else
                                 bool(alternatives) if op == "mode" else
                                 len(allocs) > 1 if op == "mod" else
                                 bool(path_lib.get((batch.origin, batch.destination))))})
    return rows


def valid_path(path, batch, tt_dict, arc_lookup):
    arcs = path.arcs
    if not arcs or path.origin != batch.origin or path.destination != batch.destination:
        return False
    nodes = [arcs[0].from_node] + [a.to_node for a in arcs]
    if (nodes[0] != batch.origin or nodes[-1] != batch.destination
            or len(set(nodes)) != len(nodes) or nodes != path.nodes
            or [a.mode for a in arcs] != path.modes):
        return False
    return all((i == 0 or arcs[i-1].to_node == arc.from_node)
               and (arc.from_node, arc.to_node, arc.mode) in arc_lookup
               and (arc.mode == "road" or bool(tt_dict.get((arc.from_node, arc.to_node, arc.mode))))
               for i, arc in enumerate(arcs))


def repair_after_mutation(ind, before, batches, path_lib, tt_dict, arc_lookup):
    """Encoding repair only. No capacity repair, route improvement, or new fallback."""
    result = dict(repair_called=True, repair_success=True, repair_action_count=0,
                  duplicate_paths_merged=0, shares_removed=0, share_normalised=False,
                  missing_allocation_restored=0, invalid_path_detected=False,
                  mutation_reverted=False)
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])
        if any(not valid_path(a.path, batch, tt_dict, arc_lookup)
               or not math.isfinite(a.share) or a.share < 0 for a in allocs):
            result.update(repair_success=False, invalid_path_detected=True, mutation_reverted=True)
            ind.od_allocations = before.od_allocations
            return result
        unique_count = len({a.path for a in allocs})
        has_duplicates = unique_count != len(allocs)
        has_filtered_share = any(a.share <= 0.05 for a in allocs)
        share_total = sum(a.share for a in allocs)
        needs_normalisation = (has_duplicates or has_filtered_share
                               or not math.isclose(share_total, 1.0,
                                                   rel_tol=0.0, abs_tol=1e-12))
        merged = allocs
        if needs_normalisation:
            old = [(a.path, a.share) for a in allocs]
            merged = base.merge_and_normalize(allocs)
            result["duplicate_paths_merged"] += len(allocs) - unique_count
            result["shares_removed"] += max(0, unique_count - len(merged))
            if old != [(a.path, a.share) for a in merged]:
                result["share_normalised"] = True
                result["repair_action_count"] += 1
                ind.od_allocations[key] = merged
        if not merged:
            options = [p for p in path_lib.get((batch.origin, batch.destination), [])
                       if valid_path(p, batch, tt_dict, arc_lookup)]
            if not options:
                result.update(repair_success=False, mutation_reverted=True)
                ind.od_allocations = before.od_allocations
                return result
            merged = [base.PathAllocation(options[0], 1.0)]
            result["missing_allocation_restored"] += 1
            result["repair_action_count"] += 1
            ind.od_allocations[key] = merged
    return result
