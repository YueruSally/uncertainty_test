"""Frozen outer instances for the post-V1 CCP100 location study.

The alpha in the old MOEA/D scenario table scales the ET-to-LT delivery
window; it is unrelated to CCP's 0.90 objective quantile.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

import baseline_uncertainty as base


SCENARIOS = {
    "S0": (20, 1.0), "S1": (10, .5), "S2": (10, 1.0),
    "S3": (10, 2.0), "S4": (20, .5), "S5": (20, 2.0),
    "S6": (30, .5), "S7": (30, 1.0), "S8": (30, 2.0),
    "S9": (50, .5), "S10": (50, 1.0), "S11": (50, 2.0),
}
VALIDATION = {"S8", "S9"}
TEST_ALPHAS = (.5, .5, 1.0, 2.0, 2.0)


def delivery_deadline(batch, alpha):
    if alpha <= 0 or batch.LT < batch.ET:
        raise ValueError("alpha must be positive and original LT must be >= ET")
    return float(batch.ET + max(alpha * (batch.LT - batch.ET), 1.0))


def _reachable(graph, origin, destination, regions, timetable_keys, max_arcs=12):
    """Deterministic structural prefilter; the normal path builder is audited too."""
    queue = deque([(origin, (origin,))])
    visited = {origin}
    while queue:
        node, nodes = queue.popleft()
        if node == destination and len(nodes) > 1:
            return True
        if len(nodes) > max_arcs:
            continue
        for next_node, arc in graph.get(node, []):
            if next_node in nodes or next_node in visited:
                continue
            if arc.mode != "road" and (arc.from_node, arc.to_node, arc.mode) not in timetable_keys:
                continue
            chain = list(nodes) + [next_node]
            if (base.china_border_monotone_ok(chain, regions)
                    and base.region_monotone_ok(chain, regions)):
                visited.add(next_node)
                queue.append((next_node, tuple(chain)))
    return False


def novel_od_pool(raw_batches, arcs, timetables, regions):
    original = {(b.origin, b.destination) for b in raw_batches}
    origins = sorted({b.origin for b in raw_batches})
    destinations = sorted({b.destination for b in raw_batches})
    tt = base.build_timetable_dict(timetables)
    graph = base.build_graph(arcs)
    return [(o, d) for o in origins for d in destinations
            if (o, d) not in original and _reachable(graph, o, d, regions, tt)]


def prepare_batches(raw_batches, k, alpha, seed, *, test=False, od_pool=()):
    """MOEA/D's batch sampling and exact deadline transform, frozen per instance."""
    if not raw_batches or k < 1:
        raise ValueError("non-empty raw batches and positive K required")
    rng = np.random.default_rng(seed)
    if not test:
        if len(raw_batches) >= k:
            indices = rng.choice(len(raw_batches), size=k, replace=False)
            selected = [deepcopy(raw_batches[int(i)]) for i in indices]
        else:
            selected = deepcopy(raw_batches)
            extra = rng.choice(len(raw_batches), size=k-len(raw_batches), replace=True)
            selected.extend(deepcopy(raw_batches[int(i)]) for i in extra)
    else:
        if not od_pool:
            raise ValueError("no structurally reachable unseen OD pairs")
        selected = []
        original_quantities = {float(b.quantity) for b in raw_batches}
        q_min, q_max = min(original_quantities), max(original_quantities)
        for i in range(k):
            o, d = od_pool[int(rng.integers(len(od_pool)))]
            template = deepcopy(raw_batches[int(rng.integers(len(raw_batches)))])
            template.origin, template.destination = o, d
            # Centi-TEU draws provide fresh quantities within the observed range.
            for _ in range(1000):
                q = round(float(rng.uniform(q_min, q_max)), 2)
                if q not in original_quantities:
                    break
            else:
                raise ValueError("could not draw an unseen batch quantity")
            template.quantity = q
            selected.append(template)
    for i, batch in enumerate(selected):
        batch.LT = delivery_deadline(batch, alpha)
        batch.batch_id = i
    return selected


def canonical_digest(value):
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def read_instance(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = manifest.pop("instance_digest")
    if canonical_digest(manifest) != digest:
        raise ValueError(f"instance digest mismatch: {path}")
    manifest["instance_digest"] = digest
    batches = [base.Batch(**row) for row in manifest["batches"]]
    if len(batches) != manifest["K"] or len({b.batch_id for b in batches}) != len(batches):
        raise ValueError("invalid instance batch count or identifiers")
    return manifest, batches


def write_catalog(data, out, seed=202600):
    if out.exists() and any(out.iterdir()):
        raise ValueError("instance directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    nodes, regions, _, _, _, arcs, ttables, raw, *_ = network
    pool = novel_od_pool(raw, arcs, ttables, regions)
    tt, lookup = base.build_timetable_dict(ttables), base.build_arc_lookup(arcs)
    import random
    viable = []
    original_state = random.getstate()
    try:
        for j, (origin, destination) in enumerate(pool):
            probe = deepcopy(raw[0])
            probe.origin, probe.destination, probe.batch_id = origin, destination, -1
            random.seed(seed + 50_000 + j)
            if base.build_path_library(nodes, regions, arcs, [probe], tt, lookup).get((origin, destination)):
                viable.append((origin, destination))
    finally:
        random.setstate(original_state)
    if not viable:
        raise ValueError("no unseen OD pair can be served by the unchanged path builder")
    # S8/S9 must not be the same batch selection as a training instance with
    # only its deadline changed: independent instance seeds avoid this leak.
    specs = [(sid, k, alpha, seed + 101 * int(sid[1:]), False)
             for sid, (k, alpha) in SCENARIOS.items()]
    specs += [(f"T{i+1}", 50, alpha, seed + 1000 + i, True)
              for i, alpha in enumerate(TEST_ALPHAS)]
    catalogue = []
    for sid, k, alpha, instance_seed, test in specs:
        batches = prepare_batches(raw, k, alpha, instance_seed, test=test, od_pool=viable)
        # Fail before optimisation if the unchanged path builder cannot serve an OD.
        state = random.getstate()
        try:
            for path_seed in range(instance_seed, instance_seed + 10):
                random.seed(path_seed)
                paths = base.build_path_library(nodes, regions, arcs, batches, tt, lookup)
                if all(paths.get((b.origin, b.destination)) for b in batches):
                    break
            else:
                raise ValueError(f"no reproducible path seed found for {sid}")
        finally:
            random.setstate(state)
        record = dict(schema_version=1, instance_id=sid,
                      split="test" if test else "validation" if sid in VALIDATION else "train",
                      K=k, deadline_window_alpha=alpha, instance_seed=instance_seed,
                      path_seed=path_seed,
                      source_data_sha256=data_sha256,
                      alpha_definition="LT=ET+max(alpha*(original_LT-ET),1 hour)",
                      new_od_required=test, new_quantity_required=test,
                      batches=[asdict(b) for b in batches])
        record["instance_digest"] = canonical_digest(record)
        (out / f"{sid}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        catalogue.append({k: record[k] for k in ("instance_id", "split", "K",
                           "deadline_window_alpha", "instance_seed", "instance_digest")})
    (out / "catalogue.json").write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    return catalogue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202600)
    args = parser.parse_args()
    write_catalog(args.data, args.out, args.seed)
