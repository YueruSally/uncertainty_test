"""Generate disjoint S0-S8 training and S0-S11 formal-test instances."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

import baseline_uncertainty as base

from .config import SCENARIOS, TEST_CONFIGS, TRAIN_CONFIGS


def canonical_digest(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _all_reachable_od(raw_batches, arcs, timetables, regions):
    origins = sorted({batch.origin for batch in raw_batches})
    destinations = sorted({batch.destination for batch in raw_batches})
    graph = base.build_graph(arcs)
    timetable = base.build_timetable_dict(timetables)
    from multi_instance_catalog import _reachable
    return [(origin, destination) for origin in origins for destination in destinations
            if origin != destination and _reachable(
                graph, origin, destination, regions, timetable)]


def _split_od_pool(pool, seed, test_fraction=0.30):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0,1)")
    ordered = sorted(pool, key=lambda item: hashlib.sha256(
        f"{seed}:{item[0]}:{item[1]}".encode()).hexdigest())
    cut = max(1, min(len(ordered) - 1, int(round(len(ordered) * (1 - test_fraction)))))
    train, test = ordered[:cut], ordered[cut:]
    if not train or not test or set(train) & set(test):
        raise ValueError("failed to create disjoint non-empty OD pools")
    return train, test


def _draw_quantity(rng, minimum, maximum, split):
    low, high = int(round(minimum * 100)), int(round(maximum * 100))
    values = np.arange(low, high + 1, dtype=int)
    parity = 0 if split == "train" else 1
    values = values[values % 2 == parity]
    if not len(values):
        raise ValueError("quantity range is too narrow for disjoint centi-TEU grids")
    return float(values[int(rng.integers(len(values)))]) / 100.0


def _draw_release_time(rng, release_time_max_h, release_time_step_h, split):
    slots = np.arange(0, int(release_time_max_h // release_time_step_h) + 1, dtype=int)
    parity = 0 if split == "train" else 1
    slots = slots[slots % 2 == parity]
    if not len(slots):
        raise ValueError("release-time range has no slots for requested split")
    return float(slots[int(rng.integers(len(slots)))] * release_time_step_h)


def _batches(raw, od_pool, k, alpha, seed, split,
             release_time_max_h, release_time_step_h):
    rng = np.random.default_rng(seed)
    quantity_min = min(float(batch.quantity) for batch in raw)
    quantity_max = max(float(batch.quantity) for batch in raw)
    windows = [float(batch.LT - batch.ET) for batch in raw if batch.LT > batch.ET]
    if not windows:
        raise ValueError("source workbook has no positive delivery windows")
    result = []
    for batch_id in range(k):
        template = deepcopy(raw[int(rng.integers(len(raw)))])
        origin, destination = od_pool[int(rng.integers(len(od_pool)))]
        release = _draw_release_time(
            rng, release_time_max_h, release_time_step_h, split)
        original_window = windows[int(rng.integers(len(windows)))]
        template.batch_id = batch_id
        template.origin, template.destination = origin, destination
        template.quantity = _draw_quantity(rng, quantity_min, quantity_max, split)
        template.ET = release
        template.LT = float(release + max(alpha * original_window, 1.0))
        result.append(template)
    return result


def _find_path_seed(seed, nodes, regions, arcs, batches, timetables, lookup):
    state = random.getstate()
    try:
        for offset in range(20):
            path_seed = seed + offset
            random.seed(path_seed)
            paths = base.build_path_library(
                nodes, regions, arcs, batches, timetables, lookup)
            if all(paths.get((batch.origin, batch.destination)) for batch in batches):
                return path_seed
    finally:
        random.setstate(state)
    raise ValueError("no reproducible path seed serves all generated batches")


def generate_catalog(data: Path, out: Path, release_time_max_h: float,
                     release_time_step_h: float, master_seed=31_000_000,
                     train_per_config=50, test_per_config=10):
    if release_time_max_h <= 0 or release_time_step_h <= 0:
        raise ValueError("explicit positive release-time range and step are required")
    if release_time_max_h < 2 * release_time_step_h:
        raise ValueError("release-time range must contain disjoint train/test slots")
    if out.exists() and any(out.iterdir()):
        raise ValueError("catalog output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    base.BORDER_EVENT_DEFINITIONS = base.load_border_event_definitions(
        base.DEFAULT_BORDER_EVENT_DATA_FILE)
    network = base.load_network_from_extended(str(data))
    nodes, regions, _, _, _, arcs, timetables, raw, *_ = network
    timetable = base.build_timetable_dict(timetables)
    lookup = base.build_arc_lookup(arcs)
    train_od, test_od = _split_od_pool(
        _all_reachable_od(raw, arcs, timetables, regions), master_seed)
    records, signatures = [], {"train": set(), "test": set()}
    tasks = [("train", sid, train_per_config) for sid in TRAIN_CONFIGS]
    tasks += [("test", sid, test_per_config) for sid in TEST_CONFIGS]
    total_instances = sum(count for _, _, count in tasks)
    print(
        f"[CATALOG] generating {total_instances} instances "
        f"(release_max={release_time_max_h}h, step={release_time_step_h}h)",
        flush=True,
    )
    instance_index = 0
    for split, sid, count in tasks:
        k, alpha = SCENARIOS[sid]
        pool = train_od if split == "train" else test_od
        for repetition in range(count):
            instance_seed = master_seed + instance_index * 100
            batches = _batches(raw, pool, k, alpha, instance_seed, split,
                               release_time_max_h, release_time_step_h)
            batch_signature = canonical_digest([asdict(batch) for batch in batches])
            if batch_signature in signatures[split]:
                raise ValueError("duplicate generated instance; change master seed")
            signatures[split].add(batch_signature)
            path_seed = _find_path_seed(
                instance_seed + 1_000_000, nodes, regions, arcs, batches,
                timetable, lookup)
            instance_id = f"{split}-{sid}-{repetition + 1:03d}"
            record = {
                "schema_version": 2, "instance_id": instance_id,
                "split": split, "configuration": sid, "K": k,
                "deadline_window_alpha": alpha,
                "alpha_definition": "LT=ET+max(alpha*sampled_source_window,1 hour)",
                "release_time_definition": {
                    "minimum_h": 0.0, "maximum_h": float(release_time_max_h),
                    "step_h": float(release_time_step_h),
                    "split_disjoint_slot_parity": True,
                },
                "quantity_definition": {
                    "minimum_teu": min(float(x.quantity) for x in raw),
                    "maximum_teu": max(float(x.quantity) for x in raw),
                    "split_disjoint_centi_teu_parity": True,
                },
                "od_pool": split, "instance_seed": instance_seed,
                "path_seed": path_seed, "ccp100_seed": 40_000_000 + instance_index,
                "oos5000_seed": (50_000_000 + instance_index if split == "test" else None),
                "source_data_sha256": data_sha, "batch_signature": batch_signature,
                "batches": [asdict(batch) for batch in batches],
            }
            record["instance_digest"] = canonical_digest(record)
            (out / f"{instance_id}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8")
            records.append({key: record[key] for key in (
                "instance_id", "split", "configuration", "K",
                "deadline_window_alpha", "instance_seed", "path_seed",
                "ccp100_seed", "oos5000_seed", "instance_digest")})
            instance_index += 1
            print(
                f"[CATALOG] {instance_index:03d}/{total_instances} "
                f"wrote {instance_id} path_seed={path_seed}",
                flush=True,
            )
    if set(train_od) & set(test_od):
        raise AssertionError("train/test OD leakage")
    catalog = {
        "schema_version": 2, "source_data_sha256": data_sha,
        "master_seed": master_seed, "train_count": len([x for x in records if x["split"] == "train"]),
        "test_count": len([x for x in records if x["split"] == "test"]),
        "train_od_pool": train_od, "test_od_pool": test_od, "instances": records,
    }
    catalog["catalog_digest"] = canonical_digest(catalog)
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"[CATALOG] complete digest={catalog['catalog_digest']}", flush=True)
    return catalog


def read_instance(path: Path):
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.pop("instance_digest")
    if canonical_digest(record) != digest:
        raise ValueError(f"instance digest mismatch: {path}")
    record["instance_digest"] = digest
    batches = [base.Batch(**row) for row in record["batches"]]
    if len(batches) != record["K"]:
        raise ValueError("instance K does not match batch count")
    return record, batches
