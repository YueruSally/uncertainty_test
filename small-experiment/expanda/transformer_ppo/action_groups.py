"""Group-aware action selection shared by training diagnostics and inference."""
from __future__ import annotations


def action_group_diagnostics(candidates, probabilities):
    """Aggregate flat candidate probabilities into declared action groups."""
    groups = [candidate.get("action_group") for candidate in candidates]
    if any(group is None for group in groups) or len(set(groups)) < 2:
        return None
    masses = {
        group: float(sum(
            probabilities[index]
            for index, candidate_group in enumerate(groups)
            if candidate_group == group))
        for group in sorted(set(groups))
    }
    safe_groups = {
        group for group, candidate in zip(groups, candidates)
        if candidate.get("safe_action") is True
    }
    safe_group = next(iter(safe_groups)) if len(safe_groups) == 1 else None
    return {
        "groups": groups,
        "masses": masses,
        "safe_group": safe_group,
    }


def grouped_argmax(candidates, probabilities, margin=0.0):
    """Choose the best group by total mass, then its best candidate.

    When a unique safe group is declared, ties and margins below ``margin``
    return that group.  Candidates without group declarations retain ordinary
    flat argmax behavior.
    """
    diagnostics = action_group_diagnostics(candidates, probabilities)
    if diagnostics is None:
        return max(range(len(probabilities)), key=lambda index: probabilities[index]), None
    masses = diagnostics["masses"]
    safe_group = diagnostics["safe_group"]
    non_safe = [group for group in masses if group != safe_group]
    if safe_group is not None and non_safe:
        best_non_safe = max(non_safe, key=lambda group: masses[group])
        selected_group = (
            best_non_safe
            if masses[best_non_safe] > masses[safe_group] + float(margin)
            else safe_group)
    else:
        selected_group = max(masses, key=lambda group: masses[group])
    eligible = [
        index for index, group in enumerate(diagnostics["groups"])
        if group == selected_group
    ]
    action = max(eligible, key=lambda index: probabilities[index])
    diagnostics = dict(diagnostics)
    diagnostics["selected_group"] = selected_group
    diagnostics["margin"] = float(margin)
    return int(action), diagnostics
