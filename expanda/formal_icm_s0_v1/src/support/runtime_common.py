import json
import hashlib
from pathlib import Path
import statistics
import numpy as np
from transformer_ppo.archive import ObjectiveNormalizer

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def _normalizer(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("source") != "training-only":
        raise ValueError("normalization must declare source=training-only")
    return ObjectiveNormalizer(
        tuple(value["minimum"]), tuple(value["maximum"]),
        tuple(value.get("reference", (1.1, 1.1, 1.1))))



def hierarchical_interval(grouped, seed):
    names = sorted(grouped)
    if len({len(grouped[n]) for n in names}) > 1:
        rng = np.random.default_rng(seed)
        outer = rng.integers(0, len(names), (50000, len(names)))
        sampled = np.zeros_like(outer, dtype=float)
        for position in range(len(names)):
            for index, name in enumerate(names):
                selected = np.flatnonzero(outer[:, position] == index)
                values = np.asarray(grouped[name], dtype=float)
                indices = rng.integers(0, len(values), (len(selected), len(values)))
                sampled[selected, position] = values[indices].mean(axis=1)
        return [float(x) for x in np.quantile(sampled.mean(axis=1), [.025, .975])]
    matrix = np.asarray([grouped[name] for name in names], dtype=float)
    generator = np.random.default_rng(seed)
    outer = generator.integers(0, matrix.shape[0],
                               (50_000, matrix.shape[0]))
    inner = generator.integers(
        0, matrix.shape[1],
        (50_000, matrix.shape[0], matrix.shape[1]))
    sampled = matrix[outer[:, :, None], inner]
    means = sampled.mean((1, 2))
    return [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]


def paired_summary(rows, key, seed):
    grouped = {}
    for row in rows:
        if row[key] is not None:
            grouped.setdefault(row["instance"], []).append(float(row[key]))
    values = [value for name in sorted(grouped) for value in grouped[name]]
    interval = hierarchical_interval(grouped, seed) if grouped else [None, None]
    return {
        "mean": statistics.fmean(statistics.fmean(grouped[name]) for name in sorted(grouped)) if grouped else None,
        "defined_pairs": len(values), "undefined_pairs": len(rows)-len(values),
        "total_pairs": len(rows),
        "hierarchical_bootstrap_95ci_low": interval[0],
        "hierarchical_bootstrap_95ci_high": interval[1],
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "per_instance": [{
            "instance": name,
            "mean": statistics.fmean(grouped[name]),
            "wins": sum(value > 1e-12 for value in grouped[name]),
            "ties": sum(abs(value) <= 1e-12 for value in grouped[name]),
            "losses": sum(value < -1e-12 for value in grouped[name]),
        } for name in sorted(grouped)],
    }
