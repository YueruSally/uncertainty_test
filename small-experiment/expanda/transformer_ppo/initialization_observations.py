"""Partial-chromosome observations for initialization path selection."""
from __future__ import annotations

from .observations import (
    PATH_BUSINESS_FEATURES, action_prior, candidate_graph_tokens,
    path_business_features, _category_code, _finite, population_summary,
)
from .graph_features import GRAPH_MODEL_SPEC, GRAPH_MODEL_SPEC_V2


INITIALIZATION_PARENT_FEATURES = (
    "batch_quantity", "share", "path_length", "arc_position", "is_arc",
    "origin_code_0", "origin_code_1", "destination_code_0",
    "destination_code_1", "from_node_code_0", "from_node_code_1",
    "to_node_code_0", "to_node_code_1", "mode_code_0", "mode_code_1",
)
INITIALIZATION_CANDIDATE_FEATURES = (
    "batch_quantity", "earliest_time", "latest_time", "selected_path_count",
    "required_path_count", "remaining_slots", "path_length", "total_distance",
    "road_fraction", "rail_fraction", "water_fraction", "other_mode_fraction",
    "origin_code_0", "origin_code_1", "destination_code_0",
    "destination_code_1", "chromosome_code_0", "chromosome_code_1",
)
INITIALIZATION_CANDIDATE_FEATURES_V2 = (
    *INITIALIZATION_CANDIDATE_FEATURES, *PATH_BUSINESS_FEATURES,
)
INITIALIZATION_CONTEXT_FEATURES_V3 = (
    "planned_share", "planned_quantity", "remaining_planned_share",
    "candidate_cost_rank", "candidate_emission_rank",
    "candidate_time_rank", "candidate_capacity_rank",
    "population_path_use_fraction", "selected_edge_novelty",
    "candidate_is_path_count", "proposed_path_count_fraction",
)
INITIALIZATION_CANDIDATE_FEATURES_V3 = (
    *INITIALIZATION_CANDIDATE_FEATURES_V2, *INITIALIZATION_CONTEXT_FEATURES_V3,
)
INITIALIZATION_GLOBAL_FEATURES = (
    "K", "deadline_window_alpha", "evaluation_fraction", "remaining_fraction",
    "initial_population_fraction", "feasible_ratio", "population_hv",
    "objective_min_0", "objective_min_1", "objective_min_2",
    "objective_span_0", "objective_span_1", "objective_span_2",
    "archive_size_fraction", "archive_hv",
)
INITIALIZATION_MODEL_SPEC = {
    "parent_dim": len(INITIALIZATION_PARENT_FEATURES),
    "candidate_dim": len(INITIALIZATION_CANDIDATE_FEATURES),
    "global_dim": len(INITIALIZATION_GLOBAL_FEATURES),
    "action_count": 1,
}
INITIALIZATION_GRAPH_MODEL_SPEC = {**INITIALIZATION_MODEL_SPEC, **GRAPH_MODEL_SPEC}
INITIALIZATION_GRAPH_MODEL_SPEC_V2 = {
    **INITIALIZATION_MODEL_SPEC,
    "candidate_dim": len(INITIALIZATION_CANDIDATE_FEATURES_V2),
    **GRAPH_MODEL_SPEC_V2,
}
INITIALIZATION_GRAPH_MODEL_SPEC_V3 = {
    **INITIALIZATION_MODEL_SPEC,
    "candidate_dim": len(INITIALIZATION_CANDIDATE_FEATURES_V3),
    **GRAPH_MODEL_SPEC_V2,
}


def _path_signature(path):
    return tuple(path.nodes), tuple(path.modes)


def _rank(values, index, higher_is_better=False):
    """Return a scale-free [0, 1] rank where zero is best and ties stay tied."""
    if len(values) <= 1:
        return 0.0
    value = values[index]
    better = sum(
        other > value if higher_is_better else other < value
        for other in values)
    return better / (len(values) - 1)


def _edge_set(path):
    return {
        (str(arc.from_node), str(arc.to_node), str(arc.mode).lower())
        for arc in path.arcs
    }


def _selected_edge_novelty(path, partial_individual, key):
    selected = [allocation.path for allocation in
                partial_individual.od_allocations.get(key, ())]
    if not selected:
        return 1.0
    candidate_edges = _edge_set(path)
    similarities = []
    for other in selected:
        other_edges = _edge_set(other)
        union = candidate_edges | other_edges
        similarities.append(
            len(candidate_edges & other_edges) / len(union) if union else 1.0)
    return 1.0 - max(similarities)


def _population_path_use(path, population, key):
    if not population:
        return 0.0
    signature = _path_signature(path)
    used = sum(any(
        _path_signature(allocation.path) == signature
        for allocation in individual.od_allocations.get(key, ()))
        for individual in population)
    return used / len(population)


def _partial_tokens(individual, batches):
    by_id = {batch.batch_id: batch for batch in batches}
    tokens = []
    for key in sorted(individual.od_allocations):
        batch = by_id.get(key[2])
        quantity = _finite(getattr(batch, "quantity", 0.0))
        origin, destination = _category_code(key[0]), _category_code(key[1])
        for allocation in individual.od_allocations[key]:
            path = allocation.path
            tokens.append([
                quantity, _finite(allocation.share), float(len(path.arcs)), 0.0, 0.0,
                *origin, *destination, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            for position, arc in enumerate(path.arcs):
                tokens.append([
                    quantity, _finite(allocation.share), float(len(path.arcs)),
                    position / max(1, len(path.arcs) - 1), 1.0,
                    *origin, *destination, *_category_code(arc.from_node),
                    *_category_code(arc.to_node), *_category_code(arc.mode),
                ])
    return tokens or [[0.0] * len(INITIALIZATION_PARENT_FEATURES)]


def _candidate_tokens(candidates, context, partial_individual, population,
                      representation="legacy"):
    batch = context["batch"]
    if context.get("decision_stage") == "path_count":
        if representation != "path_v3":
            raise ValueError("path-count control requires path_v3")
        maximum = max(1, int(context["maximum_path_count"]))
        rows = []
        for candidate in candidates:
            count = int(candidate["path_count"])
            rows.append([
                _finite(batch.quantity), _finite(batch.ET), _finite(batch.LT),
                0.0, float(count), float(count), 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                *_category_code(batch.origin), *_category_code(batch.destination),
                0.0, 0.0,
                *([0.0] * len(PATH_BUSINESS_FEATURES)),
                *([0.0] * (len(INITIALIZATION_CONTEXT_FEATURES_V3) - 2)),
                1.0, count / maximum,
            ])
        return rows, [bool(candidate["eligible"]) for candidate in candidates]
    selected_count = int(context["selection_step"])
    required_count = int(context["required_path_count"])
    key = (batch.origin, batch.destination, batch.batch_id)
    business = [path_business_features(
        candidate["path"], batch, context.get("planned_share", 0.0))
        for candidate in candidates]
    costs = [row[0] for row in business]
    emissions = [row[1] for row in business]
    times = [row[2] for row in business]
    capacities = [row[4] for row in business]
    rows = []
    for index, candidate in enumerate(candidates):
        path = candidate["path"]
        counts = {"road": 0, "rail": 0, "water": 0}
        for mode in path.modes:
            if mode in counts:
                counts[mode] += 1
        denominator = max(1, len(path.modes))
        other = denominator - sum(counts.values())
        row = [
            _finite(batch.quantity), _finite(batch.ET), _finite(batch.LT),
            float(selected_count), float(required_count),
            float(required_count - selected_count), float(len(path.arcs)),
            sum(_finite(arc.distance) for arc in path.arcs),
            counts["road"] / denominator, counts["rail"] / denominator,
            counts["water"] / denominator, other / denominator,
            *_category_code(batch.origin), *_category_code(batch.destination),
            *_category_code((tuple(path.nodes), tuple(path.modes))),
        ]
        if representation == "path_v2":
            row.extend(path_business_features(path, batch, 1.0))
        elif representation == "path_v3":
            planned_share = _finite(context.get("planned_share"))
            row.extend(business[index])
            row.extend([
                planned_share,
                _finite(batch.quantity) * planned_share,
                _finite(context.get("remaining_planned_share")),
                _rank(costs, index), _rank(emissions, index),
                _rank(times, index), _rank(capacities, index, True),
                _population_path_use(path, population, key),
                _selected_edge_novelty(path, partial_individual, key),
                0.0, required_count / max(
                    1, int(context.get("maximum_path_count", required_count))),
            ])
        elif representation != "legacy":
            raise ValueError("unknown candidate representation")
        rows.append(row)
    return rows, [bool(candidate["eligible"]) for candidate in candidates]


def build_initialization_observation(partial_individual, batches, candidates, context,
                                     population, archive_summary, normalizer,
                                     evaluation_budget, evaluation_count, pop_size,
                                     graph_observation=None,
                                     candidate_representation="legacy"):
    candidate_values, eligible = _candidate_tokens(
        candidates, context, partial_individual, population,
        candidate_representation)
    population_state = population_summary(population, normalizer)
    budget = max(1, int(evaluation_budget or 1))
    minimum = population_state["objective_min"]
    span = population_state["objective_span"]
    global_values = [
        _finite(context.get("K")), _finite(context.get("deadline_window_alpha")),
        evaluation_count / budget, max(0.0, 1.0 - evaluation_count / budget),
        len(population) / max(1, pop_size), population_state["feasible_ratio"],
        population_state["hypervolume"], *[_finite(value) for value in minimum],
        *[_finite(value) for value in span],
        _finite(archive_summary.get("archive_size")) / budget,
        _finite(archive_summary.get("archive_hv")),
    ]
    observation = {
        "parent": _partial_tokens(partial_individual, batches),
        "candidates": candidate_values,
        "eligible": eligible,
        # Base samples paths uniformly without replacement.  At each partial
        # construction step that is uniform over the remaining legal paths.
        "action_prior": action_prior(
            candidates, eligible, use_candidate_probability=False),
        "operator_id": 0,
        "global": global_values,
    }
    if graph_observation is not None:
        observation["graph"] = graph_observation
        node_index = graph_observation["node_index"]
        observation["candidate_graph_nodes"] = [
            ([node_index[name] for name in candidate["path"].nodes
              if name in node_index]
             if candidate.get("path") is not None else [])
            for candidate in candidates]
        if candidate_representation in ("path_v2", "path_v3"):
            observation["candidate_graph_tokens"] = candidate_graph_tokens(
                candidates, graph_observation)
    return observation
