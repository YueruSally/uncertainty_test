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


def candidate_tokens(candidates: Iterable[dict], operator: str, batches=()) -> tuple[list[list[float]], list[bool]]:
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
        rows.append([
            _finite(candidate.get("batch_quantity")), _finite(candidate.get("share")),
            _finite(candidate.get("path_count")),
            _finite(candidate.get("path_length")),
            _finite(candidate.get("alternative_mode_count")),
            _finite(candidate.get("available_reliable_options")),
            _finite(candidate.get("unallocated_path_count")),
            float(target.get("allocation_index") is not None),
            float(arc_index is not None), _finite(arc_index),
            *origin, *destination, *from_node, *to_node, *mode,
        ])
        eligible.append(bool(candidate.get("eligible")))
    if not rows:
        rows, eligible = [[0.0] * len(CANDIDATE_FEATURES)], [False]
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
                      evaluation_count, generations, graph_observation=None):
    operator = context["operator"]
    candidate_values, eligible = candidate_tokens(candidates, operator, batches)
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
        "operator_id": OPERATOR_TO_ID[operator],
        "global": global_values,
    }
    if graph_observation is not None:
        observation["graph"] = graph_observation
        observation["candidate_graph_nodes"] = candidate_graph_nodes(
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
    for row, item in enumerate(observations):
        p, c = len(item["parent"]), len(item["candidates"])
        parent[row, :p] = torch.as_tensor(item["parent"], dtype=torch.float32, device=device)
        candidates[row, :c] = torch.as_tensor(item["candidates"], dtype=torch.float32, device=device)
        parent_mask[row, :p] = False
        candidate_mask[row, :c] = False
        eligible[row, :c] = torch.as_tensor(item["eligible"], dtype=torch.bool, device=device)
    result = {
        "parent": parent, "candidates": candidates,
        "parent_padding_mask": parent_mask, "candidate_padding_mask": candidate_mask,
        "eligible_mask": eligible,
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
    return result
