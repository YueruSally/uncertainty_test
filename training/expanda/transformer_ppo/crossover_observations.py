"""Raw-chromosome and population observations for crossover-cut learning."""
from __future__ import annotations

from .observations import _category_code, _finite, population_summary
from .graph_features import GRAPH_MODEL_SPEC


CROSSOVER_PARENT_FEATURES = (
    "parent_side", "batch_quantity", "share", "path_length", "arc_position",
    "is_arc", "origin_code_0", "origin_code_1", "destination_code_0",
    "destination_code_1", "from_node_code_0", "from_node_code_1",
    "to_node_code_0", "to_node_code_1", "mode_code_0", "mode_code_1",
)
CROSSOVER_CANDIDATE_FEATURES = (
    "batch_quantity", "share_1", "share_2", "path_length_1", "path_length_2",
    "position_1", "position_2", "prefix_1", "suffix_1", "prefix_2",
    "suffix_2", "origin_code_0", "origin_code_1", "destination_code_0",
    "destination_code_1", "join_node_code_0", "join_node_code_1",
)
CROSSOVER_GLOBAL_FEATURES = (
    "K", "deadline_window_alpha", "generation_fraction", "evaluation_fraction",
    "remaining_fraction", "offspring_fraction", "parent_1_feasible",
    "parent_2_feasible", "parent_1_violation", "parent_2_violation",
    "feasible_ratio", "front_size_fraction", "population_hv",
    "objective_min_0", "objective_min_1", "objective_min_2",
    "objective_span_0", "objective_span_1", "objective_span_2",
    "pending_size_fraction", "pending_feasible_ratio", "pending_hv",
    "archive_size_fraction", "archive_hv",
)
CROSSOVER_MODEL_SPEC = {
    "parent_dim": len(CROSSOVER_PARENT_FEATURES),
    "candidate_dim": len(CROSSOVER_CANDIDATE_FEATURES),
    "global_dim": len(CROSSOVER_GLOBAL_FEATURES),
    "action_count": 1,
}
CROSSOVER_GRAPH_MODEL_SPEC = {**CROSSOVER_MODEL_SPEC, **GRAPH_MODEL_SPEC}


def _parent_tokens(parent, batches, side):
    by_id = {batch.batch_id: batch for batch in batches}
    tokens = []
    for key in sorted(parent.od_allocations):
        batch = by_id.get(key[2])
        quantity = _finite(getattr(batch, "quantity", 0.0))
        origin, destination = _category_code(key[0]), _category_code(key[1])
        for allocation in parent.od_allocations[key]:
            path = allocation.path
            tokens.append([
                side, quantity, _finite(allocation.share), float(len(path.arcs)),
                0.0, 0.0, *origin, *destination, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
            for position, arc in enumerate(path.arcs):
                tokens.append([
                    side, quantity, _finite(allocation.share), float(len(path.arcs)),
                    position / max(1, len(path.arcs) - 1), 1.0,
                    *origin, *destination, *_category_code(arc.from_node),
                    *_category_code(arc.to_node), *_category_code(arc.mode),
                ])
    return tokens


def _candidate_tokens(candidates):
    values = []
    for row in candidates:
        length1, length2 = row["path_length_1"], row["path_length_2"]
        position1, position2 = row["position_1"], row["position_2"]
        values.append([
            _finite(row["batch_quantity"]), _finite(row["share_1"]),
            _finite(row["share_2"]), float(length1), float(length2),
            float(position1), float(position2), float(position1),
            float(length1 - position1), float(position2),
            float(length2 - position2), *_category_code(row["origin"]),
            *_category_code(row["destination"]), *_category_code(row["join_node"]),
        ])
    return values, [bool(row["eligible"]) for row in candidates]


def build_crossover_shared_observation(
        parent1, parent2, batches, context, population, pending_offspring,
        archive_summary, normalizer, evaluation_budget, evaluation_count,
        generations, graph_observation=None):
    """Build features that stay fixed across all cuts of one crossover.

    One common-node crossover can request many Batch-level cut decisions.  The
    parents, population, pending offspring, archive, evaluation counter, and
    network graph cannot change until both children have been finalized and
    evaluated.  Computing these values once is therefore exactly equivalent to
    rebuilding them before every cut.
    """
    population_state = population_summary(population, normalizer)
    pending_state = population_summary(pending_offspring, normalizer)
    budget = max(1, int(evaluation_budget or 1))
    generation = max(0, int(context.get("generation", 0)))
    minimum = population_state["objective_min"]
    span = population_state["objective_span"]
    global_values = [
        _finite(context.get("K")), _finite(context.get("deadline_window_alpha")),
        generation / max(1, generations - 1), evaluation_count / budget,
        max(0.0, 1.0 - evaluation_count / budget),
        _finite(context.get("offspring_fraction")),
        float(bool(getattr(parent1, "feasible", False))),
        float(bool(getattr(parent2, "feasible", False))),
        _finite(getattr(parent1, "normalized_violation", 0.0)),
        _finite(getattr(parent2, "normalized_violation", 0.0)),
        population_state["feasible_ratio"],
        population_state["front_size"] / max(1, population_state["population_size"]),
        population_state["hypervolume"], *[_finite(value) for value in minimum],
        *[_finite(value) for value in span],
        pending_state["population_size"] / max(1, len(population)),
        pending_state["feasible_ratio"], pending_state["hypervolume"],
        _finite(archive_summary.get("archive_size")) / budget,
        _finite(archive_summary.get("archive_hv")),
    ]
    parent_tokens = [
        *_parent_tokens(parent1, batches, -1.0),
        *_parent_tokens(parent2, batches, 1.0),
    ]
    shared = {
        "parent": parent_tokens or [[0.0] * len(CROSSOVER_PARENT_FEATURES)],
        "operator_id": 0,
        "global": global_values,
    }
    if graph_observation is not None:
        shared["graph"] = graph_observation
    return shared


def build_crossover_observation(parent1, parent2, batches, candidates, context,
                                population, pending_offspring, archive_summary,
                                normalizer, evaluation_budget, evaluation_count,
                                generations, graph_observation=None,
                                shared_observation=None):
    candidate_values, eligible = _candidate_tokens(candidates)
    shared = shared_observation or build_crossover_shared_observation(
        parent1, parent2, batches, context, population, pending_offspring,
        archive_summary, normalizer, evaluation_budget, evaluation_count,
        generations, graph_observation)
    observation = dict(shared)
    observation["candidates"] = candidate_values
    observation["eligible"] = eligible
    if "graph" in observation:
        graph_observation = observation["graph"]
        node_index = graph_observation["node_index"]
        observation["candidate_graph_nodes"] = [
            ([node_index[row["join_node"]]]
             if row["join_node"] in node_index else [])
            for row in candidates
        ]
    return observation
