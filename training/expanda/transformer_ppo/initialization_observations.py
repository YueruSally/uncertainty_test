"""Partial-chromosome observations for initialization path selection."""
from __future__ import annotations

from .observations import _category_code, _finite, population_summary
from .graph_features import GRAPH_MODEL_SPEC


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


def _candidate_tokens(candidates, context):
    batch = context["batch"]
    selected_count = int(context["selection_step"])
    required_count = int(context["required_path_count"])
    rows = []
    for candidate in candidates:
        path = candidate["path"]
        counts = {"road": 0, "rail": 0, "water": 0}
        for mode in path.modes:
            if mode in counts:
                counts[mode] += 1
        denominator = max(1, len(path.modes))
        other = denominator - sum(counts.values())
        rows.append([
            _finite(batch.quantity), _finite(batch.ET), _finite(batch.LT),
            float(selected_count), float(required_count),
            float(required_count - selected_count), float(len(path.arcs)),
            sum(_finite(arc.distance) for arc in path.arcs),
            counts["road"] / denominator, counts["rail"] / denominator,
            counts["water"] / denominator, other / denominator,
            *_category_code(batch.origin), *_category_code(batch.destination),
            *_category_code((tuple(path.nodes), tuple(path.modes))),
        ])
    return rows, [bool(candidate["eligible"]) for candidate in candidates]


def build_initialization_observation(partial_individual, batches, candidates, context,
                                     population, archive_summary, normalizer,
                                     evaluation_budget, evaluation_count, pop_size,
                                     graph_observation=None):
    candidate_values, eligible = _candidate_tokens(candidates, context)
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
        "operator_id": 0,
        "global": global_values,
    }
    if graph_observation is not None:
        observation["graph"] = graph_observation
        node_index = graph_observation["node_index"]
        observation["candidate_graph_nodes"] = [
            [node_index[name] for name in candidate["path"].nodes
             if name in node_index]
            for candidate in candidates
        ]
    return observation
