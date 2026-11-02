"""Permutation-aware observations for mutation-location policies."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from .config import OPERATOR_TO_ID
from .archive import exact_hv_3d


PARENT_FEATURES = (
    "batch_quantity", "share",
    "arc_distance", "arc_cost_rate", "arc_emission_rate", "arc_speed",
    "arc_position", "path_length", "is_arc", "origin_code_0", "origin_code_1",
    "destination_code_0", "destination_code_1", "from_node_code_0",
    "from_node_code_1", "to_node_code_0", "to_node_code_1",
    "mode_code_0", "mode_code_1",
)
CANDIDATE_FEATURES = (
    "batch_quantity", "share", "path_count", "path_length",
    "alternative_modes", "reliable_options", "unallocated_paths",
    "allocation_present", "arc_present", "arc_position",
    "origin_code_0", "origin_code_1", "destination_code_0", "destination_code_1",
    "from_node_code_0", "from_node_code_1", "to_node_code_0", "to_node_code_1",
    "mode_code_0", "mode_code_1",
)
PATH_BUSINESS_FEATURES = (
    "path_cost_per_teu", "path_emission_per_teu", "path_nominal_time_h",
    "path_total_distance", "path_bottleneck_capacity",
    "path_road_fraction", "path_rail_fraction", "path_water_fraction",
    "path_other_mode_fraction", "path_border_arc_fraction",
    "allocated_quantity", "allocated_base_cost", "allocated_base_emission",
    "deadline_window_h", "nominal_time_slack_h",
    "arc_distance", "arc_capacity", "arc_cost_rate", "arc_emission_rate",
    "arc_speed", "arc_is_border",
)
MUTATION_OPTION_FEATURES = (
    "alternative_min_cost", "alternative_min_emission", "alternative_min_time",
    "alternative_mean_cost", "alternative_mean_emission",
    "alternative_mean_time", "reliable_best_on_time_probability",
    "reliable_mean_on_time_probability", "reliable_min_max_lateness_h",
    "reliable_mean_max_lateness_h",
)
CANDIDATE_PATH_V2_FEATURES = (
    *CANDIDATE_FEATURES, *PATH_BUSINESS_FEATURES, *MUTATION_OPTION_FEATURES,
)
MUTATION_CONTROL_FEATURES_V3 = (
    "candidate_is_operator_stage",
    "candidate_is_add", "candidate_is_del", "candidate_is_mod",
    "candidate_is_mode", "candidate_is_replace",
    "operator_legal_target_count", "operator_legal_target_fraction",
)
CANDIDATE_PATH_V3_FEATURES = (
    *CANDIDATE_PATH_V2_FEATURES, *MUTATION_CONTROL_FEATURES_V3,
)
REPLACEMENT_PATH_FEATURES_V4 = (
    "replacement_delta_cost_per_teu",
    "replacement_delta_emission_per_teu",
    "replacement_delta_nominal_time_h",
    "replacement_delta_distance",
    "replacement_delta_bottleneck_capacity",
    "replacement_on_time_probability",
    "replacement_max_lateness_h",
    "replacement_is_current_path",
)
CANDIDATE_PATH_V4_FEATURES = (
    *CANDIDATE_PATH_V3_FEATURES, *REPLACEMENT_PATH_FEATURES_V4,
)
GLOBAL_FEATURES = (
    "K", "deadline_window_alpha", "generation_fraction", "evaluation_fraction",
    "remaining_fraction", "offspring_fraction", "phase_is_boost",
    "parent_feasible", "parent_violation", "parent_rank", "parent_crowding",
    "parent_objective_0", "parent_objective_1", "parent_objective_2",
    "feasible_ratio", "front_size_fraction", "population_hv",
    "objective_min_0", "objective_min_1", "objective_min_2",
    "objective_span_0", "objective_span_1", "objective_span_2",
    "pending_size_fraction", "pending_feasible_ratio", "pending_front_size_fraction",
    "pending_hv", "pending_objective_min_0", "pending_objective_min_1",
    "pending_objective_min_2", "pending_objective_span_0",
    "pending_objective_span_1", "pending_objective_span_2",
    "archive_size_fraction", "archive_hv",
)


def _finite(value, default=0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _category_code(value) -> tuple[float, float]:
    """Stable categorical signature, never a list/order positional encoding."""
    if value is None:
        return 0.0, 0.0
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    second = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
    return 2.0 * first - 1.0, 2.0 * second - 1.0


def path_business_features(path, batch=None, share=0.0, arc=None, source=None):
    """Static path and allocation features; never calls the objective evaluator."""
    arcs = list(getattr(path, "arcs", ()) or ())
    modes = ([str(getattr(item, "mode", "")).lower() for item in arcs]
             if arcs else [str(item).lower()
                           for item in (source or {}).get("path_modes", ())])
    denominator = max(1, len(modes))
    mode_fraction = {
        name: sum(mode == name for mode in modes) / denominator
        for name in ("road", "rail", "water")
    }
    other_fraction = sum(
        mode not in {"road", "rail", "water"} for mode in modes) / denominator
    quantity = _finite(getattr(batch, "quantity", 0.0))
    allocated_quantity = quantity * _finite(share)
    source = source or {}
    cost = _finite(getattr(
        path, "base_cost_per_teu", source.get("path_cost_per_teu", 0.0)))
    emission = _finite(getattr(
        path, "base_emission_per_teu", source.get("path_emission_per_teu", 0.0)))
    nominal_time = _finite(getattr(
        path, "base_travel_time_h", source.get("path_nominal_time_h", 0.0)))
    earliest = _finite(getattr(batch, "ET", 0.0))
    latest = _finite(getattr(batch, "LT", 0.0))
    return [
        cost, emission, nominal_time,
        (sum(_finite(getattr(item, "distance", 0.0)) for item in arcs)
         if arcs else _finite(source.get("path_total_distance"))),
        (min((_finite(getattr(item, "capacity", 0.0)) for item in arcs),
             default=0.0) if arcs else _finite(source.get("path_bottleneck_capacity"))),
        mode_fraction["road"], mode_fraction["rail"], mode_fraction["water"],
        other_fraction,
        sum(bool(getattr(item, "is_border_arc", False)) for item in arcs) / denominator,
        allocated_quantity, allocated_quantity * cost,
        allocated_quantity * emission, max(0.0, latest - earliest),
        latest - earliest - nominal_time,
        _finite(getattr(arc, "distance", source.get("arc_distance", 0.0))),
        _finite(getattr(arc, "capacity", source.get("arc_capacity", 0.0))),
        _finite(getattr(
            arc, "cost_per_teu_km", source.get("arc_cost_rate", 0.0))),
        _finite(getattr(
            arc, "emission_per_teu_km", source.get("arc_emission_rate", 0.0))),
        _finite(getattr(arc, "speed_kmh", source.get("arc_speed", 0.0))),
        float(bool(getattr(arc, "is_border_arc", source.get("arc_is_border", False)))),
    ]


def action_prior(candidates, eligible, use_candidate_probability=True):
    """Return Base's candidate distribution restricted to legal actions."""
    if len(candidates) != len(eligible):
        raise ValueError("candidate prior and eligibility lengths differ")
    values = []
    for candidate, legal in zip(candidates, eligible):
        if not legal:
            values.append(0.0)
            continue
        value = (
            candidate.get("selection_probability", 1.0)
            if use_candidate_probability else 1.0)
        value = _finite(value, 0.0)
        values.append(max(0.0, value))
    mass = sum(values)
    legal_count = sum(bool(value) for value in eligible)
    if mass <= 0.0:
        return [
            (1.0 / legal_count if legal else 0.0)
            for legal in eligible
        ] if legal_count else [0.0] * len(eligible)
    return [value / mass for value in values]


def parent_tokens(ind, batches) -> list[list[float]]:
    by_id = {batch.batch_id: batch for batch in batches}
    tokens = []
    for key in sorted(ind.od_allocations):
        for alloc in ind.od_allocations[key]:
            batch = by_id.get(key[2])
            quantity = _finite(getattr(batch, "quantity", 0.0))
            path = alloc.path
            arcs = list(path.arcs)
            origin = _category_code(key[0])
            destination = _category_code(key[1])
            # Allocation token has no artificial list position.
            tokens.append([
                quantity, _finite(alloc.share),
                0.0, 0.0, 0.0, 0.0, 0.0, float(len(arcs)), 0.0,
                *origin, *destination, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            for position, arc in enumerate(arcs):
                denominator = max(1, len(arcs) - 1)
                from_node = _category_code(arc.from_node)
                to_node = _category_code(arc.to_node)
                mode = _category_code(arc.mode)
                tokens.append([
                    quantity, _finite(alloc.share),
                    _finite(arc.distance), _finite(arc.cost_per_teu_km),
                    _finite(arc.emission_per_teu_km), _finite(arc.speed_kmh),
                    position / denominator, float(len(arcs)), 1.0,
                    *origin, *destination, *from_node, *to_node, *mode,
                ])
    return tokens or [[0.0] * len(PARENT_FEATURES)]


def candidate_tokens(candidates: Iterable[dict], operator: str, batches=(),
                     representation="legacy") -> tuple[list[list[float]], list[bool]]:
    by_id = {batch.batch_id: batch for batch in batches}
    rows, eligible = [], []
    for candidate in candidates:
        target = candidate.get("target", {})
        if is_dataclass(target):
            target = asdict(target)
        arc_index = target.get("arc_index")
        batch = by_id.get(target.get("batch_id"))
        origin = _category_code(getattr(batch, "origin", None))
        destination = _category_code(getattr(batch, "destination", None))
        from_node = _category_code(candidate.get("from_node"))
        to_node = _category_code(candidate.get("to_node"))
        mode = _category_code(candidate.get("current_mode"))
        row = [
            _finite(candidate.get("batch_quantity")), _finite(candidate.get("share")),
            _finite(candidate.get("path_count")),
            _finite(candidate.get("path_length")),
            _finite(candidate.get("alternative_mode_count")),
            _finite(candidate.get("available_reliable_options")),
            _finite(candidate.get("unallocated_path_count")),
            float(target.get("allocation_index") is not None),
            float(arc_index is not None), _finite(arc_index),
            *origin, *destination, *from_node, *to_node, *mode,
        ]
        if representation in ("path_v2", "path_v3", "path_v4"):
            row.extend(path_business_features(
                candidate.get("path"), batch, candidate.get("share"),
                candidate.get("arc"), candidate))
            row.extend(_finite(candidate.get(name))
                       for name in MUTATION_OPTION_FEATURES)
            if representation in ("path_v3", "path_v4"):
                candidate_operator = candidate.get("candidate_operator", operator)
                row.append(float(bool(candidate.get("operator_stage"))))
                row.extend(
                    float(candidate_operator == name) for name in OPERATOR_TO_ID)
                row.extend([
                    _finite(candidate.get("operator_legal_target_count")),
                    _finite(candidate.get("operator_legal_target_fraction")),
                ])
                if representation == "path_v4":
                    row.extend(_finite(candidate.get(name))
                               for name in REPLACEMENT_PATH_FEATURES_V4)
        elif representation != "legacy":
            raise ValueError("unknown candidate representation")
        rows.append(row)
        eligible.append(bool(candidate.get("eligible")))
    if not rows:
        dimension = (
            len(CANDIDATE_PATH_V4_FEATURES) if representation == "path_v4" else
            len(CANDIDATE_PATH_V3_FEATURES) if representation == "path_v3" else
            len(CANDIDATE_PATH_V2_FEATURES) if representation == "path_v2" else
            len(CANDIDATE_FEATURES))
        rows, eligible = [[0.0] * dimension], [False]
    if operator not in OPERATOR_TO_ID:
        raise ValueError(f"unknown operator: {operator}")
    return rows, eligible


def candidate_graph_nodes(candidates, graph_observation):
    if graph_observation is None:
        return None
    node_index = graph_observation["node_index"]
    result = []
    for candidate in candidates:
        names = candidate.get("path_nodes") or [
            candidate.get("from_node"), candidate.get("to_node")]
        indices = [node_index[name] for name in names if name in node_index]
        result.append(indices)
    return result or [[]]


def _path_graph_tokens(nodes, modes, graph_observation, segment):
    node_index = graph_observation["node_index"]
    edge_lookup = graph_observation.get("edge_lookup", {})
    result = []
    for position, name in enumerate(nodes):
        if name in node_index:
            result.append([0, node_index[name], segment, position])
        if position < len(nodes) - 1 and position < len(modes):
            key = (nodes[position], nodes[position + 1], str(modes[position]).lower())
            if key in edge_lookup:
                result.append([1, edge_lookup[key], segment, position])
    return result


def candidate_graph_tokens(candidates, graph_observation):
    """Ordered alternating node/edge references for each candidate path."""
    if graph_observation is None:
        return None
    result = []
    for candidate in candidates:
        paths = candidate.get("candidate_graph_paths")
        path = candidate.get("path")
        if paths is None and path is not None:
            paths = [(list(path.nodes), list(path.modes))]
        if paths is None:
            nodes = list(candidate.get("path_nodes") or [])
            modes = list(candidate.get("path_modes") or [])
            if not nodes:
                nodes = [name for name in (
                    candidate.get("from_node"), candidate.get("to_node")) if name]
                modes = [candidate.get("current_mode")] if len(nodes) == 2 else []
            paths = [(nodes, modes)]
        tokens = []
        for segment, (nodes, modes) in enumerate(paths):
            tokens.extend(_path_graph_tokens(
                list(nodes), list(modes), graph_observation, min(segment, 1)))
        result.append(tokens)
    return result or [[]]


def population_summary(population, normalizer=None) -> dict:
    if not population:
        return {"population_size": 0, "feasible_ratio": 0.0, "front_size": 0,
                "objective_min": [0.0] * 3, "objective_span": [0.0] * 3,
                "hypervolume": 0.0}
    feasible = [ind for ind in population if getattr(ind, "feasible", False)]
    front = [ind for ind in feasible if getattr(ind, "rank", None) == 0]
    source = feasible or population
    values = np.asarray([ind.objectives for ind in source], dtype=float)
    finite_rows = values[np.all(np.isfinite(values), axis=1)]
    if len(finite_rows):
        minimum, maximum = finite_rows.min(axis=0), finite_rows.max(axis=0)
    else:
        minimum = maximum = np.zeros(3, dtype=float)
    hv = 0.0
    if normalizer is not None:
        normalized = []
        for individual in feasible:
            try:
                normalized.append(normalizer.transform(individual.objectives))
            except ValueError:
                pass
        hv = exact_hv_3d(normalized, normalizer.reference)
    return {"population_size": len(population), "feasible_ratio": len(feasible) / len(population),
            "front_size": len(front), "objective_min": np.nan_to_num(minimum).tolist(),
            "objective_span": np.nan_to_num(maximum - minimum).tolist(),
            "hypervolume": hv}


def build_observation(ind, batches, candidates, context, population, pending_offspring,
                      archive_summary, normalizer, evaluation_budget,
                      evaluation_count, generations, graph_observation=None,
                      candidate_representation="legacy"):
    operator = context["operator"]
    candidate_values, eligible = candidate_tokens(
        candidates, operator, batches, candidate_representation)
    pop = population_summary(population, normalizer)
    pending = population_summary(pending_offspring, normalizer)
    budget = max(1, int(evaluation_budget or 1))
    generation = max(0, int(context.get("generation", 0)))
    mins, spans = pop["objective_min"], pop["objective_span"]
    pending_mins, pending_spans = pending["objective_min"], pending["objective_span"]
    global_values = [
        _finite(context.get("K")), _finite(context.get("deadline_window_alpha")),
        generation / max(1, generations - 1), evaluation_count / budget,
        max(0.0, 1.0 - evaluation_count / budget),
        _finite(context.get("offspring_fraction")), float(context.get("phase") == "boost"),
        float(bool(context.get("feasible_before"))), _finite(context.get("violation_before")),
        _finite(context.get("parent_rank"), -1.0), _finite(context.get("parent_crowding")),
        _finite(context.get("q90_cost_before")), _finite(context.get("q90_emission_before")),
        _finite(context.get("q90_makespan_before")), pop["feasible_ratio"],
        pop["front_size"] / max(1, pop["population_size"]), pop["hypervolume"],
        *[_finite(v) for v in mins], *[_finite(v) for v in spans],
        pending["population_size"] / max(1, len(population)), pending["feasible_ratio"],
        pending["front_size"] / max(1, pending["population_size"]), pending["hypervolume"],
        *[_finite(v) for v in pending_mins], *[_finite(v) for v in pending_spans],
        _finite(archive_summary.get("archive_size")) / max(1, budget),
        _finite(archive_summary.get("archive_hv")),
    ]
    observation = {
        "parent": parent_tokens(ind, batches),
        "candidates": candidate_values,
        "eligible": eligible,
        "action_prior": action_prior(candidates, eligible),
        "operator_id": OPERATOR_TO_ID[operator],
        "global": global_values,
    }
    if graph_observation is not None:
        observation["graph"] = graph_observation
        observation["candidate_graph_nodes"] = candidate_graph_nodes(
            candidates, graph_observation)
        if candidate_representation in ("path_v2", "path_v3", "path_v4"):
            observation["candidate_graph_tokens"] = candidate_graph_tokens(
                candidates, graph_observation)
    return observation


def pad_observations(observations, device=None):
    """Convert variable token lists to padded tensors; torch is imported lazily."""
    import torch

    batch = len(observations)
    if not batch:
        raise ValueError("at least one observation is required")
    parent_dim = len(observations[0]["parent"][0])
    candidate_dim = len(observations[0]["candidates"][0])
    global_dim = len(observations[0]["global"])
    if any(any(len(token) != parent_dim for token in item["parent"])
           or any(len(token) != candidate_dim for token in item["candidates"])
           or len(item["global"]) != global_dim for item in observations):
        raise ValueError("observation feature dimensions must be consistent")
    max_parent = max(len(item["parent"]) for item in observations)
    max_candidates = max(len(item["candidates"]) for item in observations)
    parent = torch.zeros(batch, max_parent, parent_dim, dtype=torch.float32, device=device)
    candidates = torch.zeros(
        batch, max_candidates, candidate_dim, dtype=torch.float32, device=device)
    parent_mask = torch.ones(batch, max_parent, dtype=torch.bool, device=device)
    candidate_mask = torch.ones(batch, max_candidates, dtype=torch.bool, device=device)
    eligible = torch.zeros(batch, max_candidates, dtype=torch.bool, device=device)
    action_prior_values = torch.zeros(
        batch, max_candidates, dtype=torch.float32, device=device)
    for row, item in enumerate(observations):
        p, c = len(item["parent"]), len(item["candidates"])
        parent[row, :p] = torch.as_tensor(item["parent"], dtype=torch.float32, device=device)
        candidates[row, :c] = torch.as_tensor(item["candidates"], dtype=torch.float32, device=device)
        parent_mask[row, :p] = False
        candidate_mask[row, :c] = False
        eligible[row, :c] = torch.as_tensor(item["eligible"], dtype=torch.bool, device=device)
        prior = item.get("action_prior")
        if prior is None:
            prior = action_prior(
                [{} for _ in item["candidates"]], item["eligible"],
                use_candidate_probability=False)
        if len(prior) != c:
            raise ValueError("action prior must match candidate count")
        prior = action_prior(
            [{"selection_probability": value} for value in prior],
            item["eligible"])
        action_prior_values[row, :c] = torch.as_tensor(
            prior, dtype=torch.float32, device=device)
    result = {
        "parent": parent, "candidates": candidates,
        "parent_padding_mask": parent_mask, "candidate_padding_mask": candidate_mask,
        "eligible_mask": eligible,
        "action_prior": action_prior_values,
        "operator_id": torch.as_tensor([x["operator_id"] for x in observations],
                                       dtype=torch.long, device=device),
        "global": torch.as_tensor([x["global"] for x in observations],
                                   dtype=torch.float32, device=device),
    }
    graph_presence = ["graph" in item for item in observations]
    if any(graph_presence) and not all(graph_presence):
        raise ValueError("graph observations cannot be mixed with graph-free observations")
    if all(graph_presence):
        node_dim = len(observations[0]["graph"]["nodes"][0])
        edge_rows = [item["graph"]["edges"] for item in observations]
        nonempty_edges = [edge for rows in edge_rows for edge in rows]
        edge_dim = (len(nonempty_edges[0]) if nonempty_edges
                    else int(observations[0]["graph"].get("edge_dim", 0)))
        if edge_dim < 1:
            raise ValueError("graph observation must declare at least one edge feature")
        if any(
                any(len(row) != node_dim for row in item["graph"]["nodes"])
                or any(len(row) != edge_dim for row in item["graph"]["edges"])
                or len(item["graph"]["edges"]) != len(item["graph"]["edge_index"])
                for item in observations):
            raise ValueError("graph observation dimensions must be consistent")
        max_nodes = max(len(item["graph"]["nodes"]) for item in observations)
        max_edges = max(1, max(len(item["graph"]["edges"]) for item in observations))
        graph_nodes = torch.zeros(
            batch, max_nodes, node_dim, dtype=torch.float32, device=device)
        graph_edges = torch.zeros(
            batch, max_edges, edge_dim, dtype=torch.float32, device=device)
        graph_node_mask = torch.ones(
            batch, max_nodes, dtype=torch.bool, device=device)
        graph_edge_mask = torch.ones(
            batch, max_edges, dtype=torch.bool, device=device)
        graph_edge_index = torch.zeros(
            batch, max_edges, 2, dtype=torch.long, device=device)
        for row, item in enumerate(observations):
            graph = item["graph"]
            n, e = len(graph["nodes"]), len(graph["edges"])
            graph_nodes[row, :n] = torch.as_tensor(
                graph["nodes"], dtype=torch.float32, device=device)
            graph_node_mask[row, :n] = False
            if e:
                graph_edges[row, :e] = torch.as_tensor(
                    graph["edges"], dtype=torch.float32, device=device)
                graph_edge_index[row, :e] = torch.as_tensor(
                    graph["edge_index"], dtype=torch.long, device=device)
                graph_edge_mask[row, :e] = False
        result.update({
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_node_padding_mask": graph_node_mask,
            "graph_edge_padding_mask": graph_edge_mask,
            "graph_edge_index": graph_edge_index,
        })
        candidate_graph_rows = [
            item.get("candidate_graph_nodes", [[] for _ in item["candidates"]])
            for item in observations]
        if any(len(rows) != len(item["candidates"])
               for rows, item in zip(candidate_graph_rows, observations)):
            raise ValueError("candidate graph-node rows must match candidate actions")
        max_path_nodes = max(
            1, max((len(indices) for rows in candidate_graph_rows for indices in rows),
                   default=0))
        candidate_graph_index = torch.zeros(
            batch, max_candidates, max_path_nodes, dtype=torch.long, device=device)
        candidate_graph_mask = torch.ones(
            batch, max_candidates, max_path_nodes, dtype=torch.bool, device=device)
        for row, rows in enumerate(candidate_graph_rows):
            for candidate_index, indices in enumerate(rows):
                if indices:
                    size = len(indices)
                    candidate_graph_index[row, candidate_index, :size] = torch.as_tensor(
                        indices, dtype=torch.long, device=device)
                    candidate_graph_mask[row, candidate_index, :size] = False
        result.update({
            "candidate_graph_node_index": candidate_graph_index,
            "candidate_graph_node_padding_mask": candidate_graph_mask,
        })
        token_presence = ["candidate_graph_tokens" in item for item in observations]
        if any(token_presence) and not all(token_presence):
            raise ValueError(
                "ordered candidate graph tokens cannot be mixed with legacy rows")
        if all(token_presence):
            token_rows = [item["candidate_graph_tokens"] for item in observations]
            if any(len(rows) != len(item["candidates"])
                   for rows, item in zip(token_rows, observations)):
                raise ValueError(
                    "candidate graph-token rows must match candidate actions")
            max_tokens = max(
                1, max((len(tokens) for rows in token_rows for tokens in rows),
                       default=0))
            token_index = torch.zeros(
                batch, max_candidates, max_tokens, dtype=torch.long, device=device)
            token_type = torch.zeros_like(token_index)
            token_segment = torch.zeros_like(token_index)
            token_position = torch.zeros(
                batch, max_candidates, max_tokens, dtype=torch.float32,
                device=device)
            token_mask = torch.ones(
                batch, max_candidates, max_tokens, dtype=torch.bool, device=device)
            for row, rows in enumerate(token_rows):
                for candidate_index, tokens in enumerate(rows):
                    for token_offset, token in enumerate(tokens):
                        if len(token) != 4:
                            raise ValueError(
                                "candidate graph token must have four fields")
                        kind, index, segment, position = token
                        token_type[row, candidate_index, token_offset] = int(kind)
                        token_index[row, candidate_index, token_offset] = int(index)
                        token_segment[row, candidate_index, token_offset] = int(segment)
                        token_position[row, candidate_index, token_offset] = float(position)
                        token_mask[row, candidate_index, token_offset] = False
            result.update({
                "candidate_graph_token_index": token_index,
                "candidate_graph_token_type": token_type,
                "candidate_graph_token_segment": token_segment,
                "candidate_graph_token_position": token_position,
                "candidate_graph_token_padding_mask": token_mask,
            })
    return result
