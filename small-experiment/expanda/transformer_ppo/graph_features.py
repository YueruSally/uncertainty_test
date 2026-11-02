"""Shared, deterministic network-graph observations for I/C/M policies."""
from __future__ import annotations

import hashlib
import math


GRAPH_NODE_FEATURES = (
    "hold_cost", "processing_cost", "transfer_cost",
    "in_degree", "out_degree", "origin_batch_quantity",
    "destination_batch_quantity", "origin_batch_count",
    "destination_batch_count", "is_batch_terminal",
    "region_code_0", "region_code_1",
)
GRAPH_EDGE_FEATURES = (
    "distance", "capacity", "cost_rate", "emission_rate", "speed",
    "is_border", "mode_road", "mode_rail", "mode_water", "mode_other",
)
GRAPH_EDGE_FEATURES_V2 = (
    *GRAPH_EDGE_FEATURES,
    "scheduled_service_count", "frequency_per_week", "first_departure_hour",
    "headway_hours", "timetable_travel_time_h",
)
GRAPH_MODEL_SPEC = {
    "graph_node_dim": len(GRAPH_NODE_FEATURES),
    "graph_edge_dim": len(GRAPH_EDGE_FEATURES),
}
GRAPH_MODEL_SPEC_V2 = {
    "graph_node_dim": len(GRAPH_NODE_FEATURES),
    "graph_edge_dim": len(GRAPH_EDGE_FEATURES_V2),
}


def _finite(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _category_code(value):
    if value is None:
        return 0.0, 0.0
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    second = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
    return 2.0 * first - 1.0, 2.0 * second - 1.0


def _scale(values):
    maximum = max((abs(_finite(value)) for value in values), default=0.0)
    return maximum if maximum > 0.0 else 1.0


def build_graph_observation(nodes, regions, hold_cost, processing_cost,
                            transfer_cost, arcs, batches, timetables=(),
                            representation="legacy"):
    """Encode one instance graph without using objectives or test statistics.

    The ordering and scaling rules are shared by all three learning components.
    Continuous columns are divided by an instance-local positive maximum; this
    is deterministic network/instance preprocessing, not fitted train/test
    normalization.
    """
    node_names = sorted(set(nodes)
                        | {arc.from_node for arc in arcs}
                        | {arc.to_node for arc in arcs}
                        | {batch.origin for batch in batches}
                        | {batch.destination for batch in batches})
    if not node_names:
        raise ValueError("network graph must contain at least one node")
    node_index = {name: index for index, name in enumerate(node_names)}

    incoming = {name: 0 for name in node_names}
    outgoing = {name: 0 for name in node_names}
    origin_quantity = {name: 0.0 for name in node_names}
    destination_quantity = {name: 0.0 for name in node_names}
    origin_count = {name: 0 for name in node_names}
    destination_count = {name: 0 for name in node_names}
    for arc in arcs:
        outgoing[arc.from_node] += 1
        incoming[arc.to_node] += 1
    for batch in batches:
        quantity = _finite(batch.quantity)
        origin_quantity[batch.origin] += quantity
        destination_quantity[batch.destination] += quantity
        origin_count[batch.origin] += 1
        destination_count[batch.destination] += 1

    node_columns = [
        [_finite(hold_cost.get(name)) for name in node_names],
        [_finite(processing_cost.get(name)) for name in node_names],
        [_finite(transfer_cost.get(name)) for name in node_names],
        [incoming[name] for name in node_names],
        [outgoing[name] for name in node_names],
        [origin_quantity[name] for name in node_names],
        [destination_quantity[name] for name in node_names],
        [origin_count[name] for name in node_names],
        [destination_count[name] for name in node_names],
    ]
    node_scales = [_scale(column) for column in node_columns]
    node_values = []
    for position, name in enumerate(node_names):
        numerical = [column[position] / scale
                     for column, scale in zip(node_columns, node_scales)]
        terminal = float(bool(origin_count[name] or destination_count[name]))
        node_values.append([
            *numerical, terminal, *_category_code(regions.get(name)),
        ])

    if representation not in ("legacy", "path_v2", "path_v3", "path_v4"):
        raise ValueError("unknown graph representation")
    ordered_path_representation = representation in (
        "path_v2", "path_v3", "path_v4")
    ordered_arcs = sorted(
        arcs, key=lambda arc: (
            str(arc.from_node), str(arc.to_node), str(arc.mode),
            _finite(arc.distance), _finite(arc.capacity)))
    edge_columns = [
        [_finite(arc.distance) for arc in ordered_arcs],
        [_finite(arc.capacity) for arc in ordered_arcs],
        [_finite(arc.cost_per_teu_km) for arc in ordered_arcs],
        [_finite(arc.emission_per_teu_km) for arc in ordered_arcs],
        [_finite(arc.speed_kmh) for arc in ordered_arcs],
    ]
    edge_scales = [_scale(column) for column in edge_columns]
    timetable_by_arc = {}
    for entry in timetables:
        key = (entry.from_node, entry.to_node, str(entry.mode).lower())
        timetable_by_arc.setdefault(key, []).append(entry)
    timetable_columns = []
    for arc in ordered_arcs:
        entries = timetable_by_arc.get(
            (arc.from_node, arc.to_node, str(arc.mode).lower()), [])
        timetable_columns.append([
            len(entries),
            sum(_finite(getattr(item, "frequency_per_week", 0.0))
                for item in entries),
            min((_finite(getattr(item, "first_departure_hour", 0.0))
                 for item in entries),
                default=0.0),
            min((_finite(getattr(item, "headway_hours", 0.0))
                 for item in entries), default=0.0),
            min((_finite(getattr(item, "travel_time_h", 0.0))
                 for item in entries
                 if getattr(item, "travel_time_h", None) is not None),
                default=0.0),
        ])
    timetable_scales = [
        _scale([row[column] for row in timetable_columns])
        for column in range(5)
    ]
    edge_values, edge_index, edge_lookup = [], [], {}
    for position, arc in enumerate(ordered_arcs):
        numerical = [column[position] / scale
                     for column, scale in zip(edge_columns, edge_scales)]
        mode = str(arc.mode).lower()
        values = [
            *numerical, float(bool(arc.is_border_arc)),
            float(mode == "road"), float(mode == "rail"),
            float(mode == "water"), float(mode not in {"road", "rail", "water"}),
        ]
        if ordered_path_representation:
            values.extend(
                value / scale for value, scale in
                zip(timetable_columns[position], timetable_scales))
        edge_values.append(values)
        edge_index.append([node_index[arc.from_node], node_index[arc.to_node]])
        edge_lookup[(arc.from_node, arc.to_node, mode)] = position

    return {
        "schema": (
            "ordered-node-edge-paths-v2"
            if ordered_path_representation else "shared-network-tokens-v1"),
        "nodes": node_values,
        "edges": edge_values,
        "edge_index": edge_index,
        "edge_dim": (len(GRAPH_EDGE_FEATURES_V2)
                     if ordered_path_representation else len(GRAPH_EDGE_FEATURES)),
        "node_index": node_index,
        "edge_lookup": edge_lookup,
    }
