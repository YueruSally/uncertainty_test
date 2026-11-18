"""Mask positions before selection without changing the five mutation operators."""
import math

import baseline_uncertainty as base
from learning_mutation import enumerate_targets
from learning_mutation import MutationTarget
from multi_instance_policy import rule_probabilities


def candidates_for(ind, batches, operator, path_lib, timetable, arc_lookup, reliable_options):
    rows = enumerate_targets(ind, batches, operator, path_lib, timetable, arc_lookup)
    by_id = {b.batch_id: b for b in batches}
    for row in rows:
        batch = by_id[row["target"].batch_id]
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = ind.od_allocations.get(key, [])
        paths = path_lib.get((batch.origin, batch.destination), [])
        unallocated = [p for p in paths if p not in {a.path for a in allocs}]
        row["unallocated_path_count"] = len(unallocated)
        for name, attribute in (
                ("cost", "base_cost_per_teu"),
                ("emission", "base_emission_per_teu"),
                ("time", "base_travel_time_h")):
            values = [float(getattr(path, attribute)) for path in unallocated]
            row[f"alternative_min_{name}"] = min(values, default=0.0)
            row[f"alternative_mean_{name}"] = (
                sum(values) / len(values) if values else 0.0)
        options = (reliable_options or {}).get(key, [])
        row["available_reliable_options"] = len(options)
        probabilities = [float(option.on_time_probability) for option in options]
        finite_lateness = [float(option.max_lateness_h) for option in options
                           if math.isfinite(float(option.max_lateness_h))]
        row["reliable_best_on_time_probability"] = max(probabilities, default=0.0)
        row["reliable_mean_on_time_probability"] = (
            sum(probabilities) / len(probabilities) if probabilities else 0.0)
        row["reliable_min_max_lateness_h"] = min(finite_lateness, default=0.0)
        row["reliable_mean_max_lateness_h"] = (
            sum(finite_lateness) / len(finite_lateness) if finite_lateness else 0.0)
        if operator == "replace":
            if not options:
                row["eligible"] = False
                reason = "no_reliable_option"
            elif len(allocs) == 1 and allocs[0].share == 1.0 and all(
                    o.path == allocs[0].path for o in options):
                row["eligible"] = False
                reason = "same_only_replacement"
            else:
                row["eligible"] = True
                reason = None
        elif operator == "add" and not row["unallocated_path_count"]:
            # Existing add fallback can execute but replaces an existing path;
            # it is not an actual addition. Mask this guaranteed no-add case.
            row["eligible"] = False
            reason = "no_new_path"
        elif not row["eligible"]:
            reason = {"del": "single_or_no_path", "mod": "share_cannot_change",
                      "mode": "no_alternative_mode", "add": "no_path_library"}.get(operator,
                                                                                "structural")
        else:
            reason = None
        row["mask_reason"] = reason
    rule = rule_probabilities(rows)
    for row, p in zip(rows, rule):
        row["p_rule"] = p
    return rows


def replacement_path_candidates(ind, batches, reliable_options):
    """Enumerate the joint (batch, replacement path) action used by V5-R."""
    rows = []
    batch_count = max(1, len(batches))
    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocations = list(ind.od_allocations.get(key, ()))
        current_paths = {allocation.path for allocation in allocations}
        options = list((reliable_options or {}).get(key, ()))
        alternatives = [option for option in options
                        if option.path not in current_paths]
        selectable = alternatives or options
        if not selectable:
            rows.append({
                "target": MutationTarget(batch.batch_id),
                "selection_probability": 1.0 / batch_count,
                "batch_quantity": float(batch.quantity),
                "path_count": len(allocations),
                "share": 1.0,
                "available_reliable_options": 0,
                "eligible": False,
                "mask_reason": "no_reliable_option",
                "candidate_operator": "replace",
            })
            continue

        quantity = float(batch.quantity)
        old_cost = sum(float(allocation.share)
                       * float(allocation.path.base_cost_per_teu)
                       for allocation in allocations)
        old_emission = sum(float(allocation.share)
                           * float(allocation.path.base_emission_per_teu)
                           for allocation in allocations)
        old_time = sum(float(allocation.share)
                       * float(allocation.path.base_travel_time_h)
                       for allocation in allocations)
        old_distance = sum(
            float(allocation.share)
            * sum(float(arc.distance) for arc in allocation.path.arcs)
            for allocation in allocations)
        old_capacity = sum(
            float(allocation.share)
            * min((float(arc.capacity) for arc in allocation.path.arcs), default=0.0)
            for allocation in allocations)
        same_only = not alternatives
        probability = 1.0 / batch_count / len(selectable)
        current_graph_path = (
            (list(allocations[0].path.nodes), list(allocations[0].path.modes))
            if allocations else ([], []))
        for option in selectable:
            path = option.path
            new_distance = sum(float(arc.distance) for arc in path.arcs)
            new_capacity = min(
                (float(arc.capacity) for arc in path.arcs), default=0.0)
            rows.append({
                "target": MutationTarget(
                    batch.batch_id,
                    replacement_path_id=str(path.path_id)),
                "selection_probability": probability,
                "batch_quantity": quantity,
                "path_count": len(allocations),
                "share": 1.0,
                "path": path,
                "path_id": path.path_id,
                "path_nodes": list(path.nodes),
                "path_modes": list(path.modes),
                "path_length": len(path.arcs),
                "path_cost_per_teu": float(path.base_cost_per_teu),
                "path_emission_per_teu": float(path.base_emission_per_teu),
                "path_nominal_time_h": float(path.base_travel_time_h),
                "path_total_distance": new_distance,
                "path_bottleneck_capacity": new_capacity,
                "available_reliable_options": len(options),
                "replacement_delta_cost_per_teu": (
                    float(path.base_cost_per_teu) - old_cost),
                "replacement_delta_emission_per_teu": (
                    float(path.base_emission_per_teu) - old_emission),
                "replacement_delta_nominal_time_h": (
                    float(path.base_travel_time_h) - old_time),
                "replacement_delta_distance": new_distance - old_distance,
                "replacement_delta_bottleneck_capacity": (
                    new_capacity - old_capacity),
                "replacement_on_time_probability": float(
                    option.on_time_probability),
                "replacement_max_lateness_h": float(option.max_lateness_h),
                "replacement_is_current_path": float(path in current_paths),
                "candidate_graph_paths": [
                    current_graph_path,
                    (list(path.nodes), list(path.modes)),
                ],
                "eligible": not same_only,
                "mask_reason": "same_only_replacement" if same_only else None,
                "candidate_operator": "replace",
            })
    return rows
