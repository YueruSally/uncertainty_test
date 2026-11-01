"""Pure Rule-I path scoring helpers with no controller dependencies."""
from __future__ import annotations


FEATURES = ("base_cost_per_teu", "base_emission_per_teu", "base_travel_time_h")


def validate_rule_normalization(value):
    if (value.get("source") != "training-only"
            or value.get("configuration") != "S0"
            or value.get("features") != list(FEATURES)
            or value.get("test_instances_used") is not False):
        raise ValueError("invalid training-only Rule-I path normalization")
    minimum = tuple(float(item) for item in value["minimum"])
    maximum = tuple(float(item) for item in value["maximum"])
    if len(minimum) != 3 or len(maximum) != 3 or any(
            high <= low for low, high in zip(minimum, maximum)):
        raise ValueError("Rule-I path normalization ranges must be positive")
    return minimum, maximum


def path_rule_score(path, normalization):
    minimum, maximum = validate_rule_normalization(normalization)
    values = tuple(float(getattr(path, feature)) for feature in FEATURES)
    return sum(
        (value - low) / (high - low)
        for value, low, high in zip(values, minimum, maximum)
    )
