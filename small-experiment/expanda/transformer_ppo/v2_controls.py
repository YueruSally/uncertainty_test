"""Version-2 controls that do not perturb NSGA-II's random stream."""
from __future__ import annotations

import hashlib


def validate_model_choice_probability(value):
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("model choice probability must be in [0, 1]")
    return value


def private_seed(*parts):
    """Return a stable seed owned by a control, never by the optimizer."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)
