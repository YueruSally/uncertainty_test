"""Pre-mutation features shared by logging, fitting and inference."""
from dataclasses import asdict, is_dataclass

import pandas as pd


NUMERIC = [
    "K", "deadline_window_alpha", "batch_id", "allocation_index", "arc_index",
    "p_original", "p_rule", "batch_quantity", "path_count", "share",
    "path_cost_per_teu", "path_emission_per_teu", "path_nominal_time_h",
    "alternative_mode_count", "available_reliable_options",
    "unallocated_path_count", "generation", "feasible_before",
    "q90_cost_before", "q90_emission_before", "q90_makespan_before",
    "violation_before",
]
CATEGORICAL = ["operator", "phase", "from_node", "to_node", "current_mode"]
FEATURES = NUMERIC + CATEGORICAL


def feature_record(candidate, context):
    target = candidate.get("target", candidate)
    if is_dataclass(target):
        target = asdict(target)
    values = {
        "K": context.get("K"),
        "deadline_window_alpha": context.get("deadline_window_alpha"),
        "batch_id": target.get("batch_id"),
        "allocation_index": target.get("allocation_index"),
        "arc_index": target.get("arc_index"),
        "p_original": candidate.get("selection_probability", candidate.get("p_original")),
        "p_rule": candidate.get("p_rule"),
        "batch_quantity": candidate.get("batch_quantity"),
        "path_count": candidate.get("path_count"),
        "share": candidate.get("share"),
        "path_cost_per_teu": candidate.get("path_cost_per_teu"),
        "path_emission_per_teu": candidate.get("path_emission_per_teu"),
        "path_nominal_time_h": candidate.get("path_nominal_time_h"),
        "alternative_mode_count": candidate.get("alternative_mode_count"),
        "available_reliable_options": candidate.get("available_reliable_options"),
        "unallocated_path_count": candidate.get("unallocated_path_count"),
        "generation": context.get("generation"),
        "feasible_before": int(bool(context.get("feasible_before"))),
        "q90_cost_before": context.get("q90_cost_before"),
        "q90_emission_before": context.get("q90_emission_before"),
        "q90_makespan_before": context.get("q90_makespan_before"),
        "violation_before": context.get("violation_before"),
        "operator": context.get("operator"), "phase": context.get("phase"),
        "from_node": candidate.get("from_node"),
        "to_node": candidate.get("to_node"),
        "current_mode": candidate.get("current_mode"),
    }
    return values


def feature_frame(records):
    return pd.DataFrame(records, columns=FEATURES)
