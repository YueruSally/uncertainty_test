"""Outcome-independent selection of pre-action mutation states."""


PHASES = ("early", "middle", "late")


def validate_policy_state_ids(values, state_count):
    if values is None:
        return None
    values = list(values)
    if (state_count <= 0 or len(values) != state_count
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values)
            or values != sorted(set(values))):
        raise ValueError("audit policy-state ids must be sorted unique positive integers, one per requested state")
    return values


def select_phase_states(opportunities):
    """Target 10/11/11; exhaust scarce phases, redistribute without replacement.

    Input values are ordered pre-action policy-state ids, never rewards. There
    must be at least one opportunity in each phase and at least 32 overall.
    """
    counts = {phase: len(opportunities[phase]) for phase in PHASES}
    if min(counts.values()) < 1 or sum(counts.values()) < 32:
        raise ValueError(f"trajectory lacks phase coverage or 32 distinct opportunities: {counts}")
    quotas = {phase: min(goal, counts[phase]) for phase, goal in zip(PHASES, (10, 11, 11))}
    while sum(quotas.values()) < 32:
        for phase in PHASES:
            if quotas[phase] < counts[phase] and sum(quotas.values()) < 32:
                quotas[phase] += 1
    selected = []
    for phase in PHASES:
        ids, count = opportunities[phase], quotas[phase]
        # Deterministic midpoint quantiles cover the entire available phase.
        selected.extend(ids[((2 * index + 1) * len(ids)) // (2 * count)] for index in range(count))
    selected = sorted(selected)
    validate_policy_state_ids(selected, 32)
    return {"available_by_phase": counts, "quotas": quotas,
            "policy_state_ids": selected,
            "capacity_limited_phases": [phase for phase, goal in zip(PHASES, (10, 11, 11)) if counts[phase] < goal]}
