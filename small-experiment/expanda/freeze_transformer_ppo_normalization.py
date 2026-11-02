#!/usr/bin/env python3
"""Freeze reward-HV bounds from training-only Rule optimization fronts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from multi_instance_logging import write_json
from transformer_ppo.instance_catalog import read_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Root containing training-only Rule run directories.")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--pad-fraction", type=float, default=0.10)
    parser.add_argument("--expected-instances", type=int, default=450,
                        help="Predeclared training-instance count; formal default is 450.")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("normalization output already exists")
    points, source_files, seen_instances = [], [], set()
    configurations = Counter()
    for path in sorted(args.runs.rglob("final_feasible_nondominated.json")):
        config_path = path.parent / "configuration.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get("method") not in ("Rule-CCP100", "rule-masked-random-selection"):
            continue
        instance_path = args.instances / f"{config['instance_id']}.json"
        instance, batches = read_instance(instance_path)
        if instance["split"] != "train":
            raise ValueError(f"non-training run supplied: {path}")
        if instance["instance_id"] in seen_instances:
            raise ValueError(f"multiple Rule fronts supplied for {instance['instance_id']}")
        seen_instances.add(instance["instance_id"])
        configurations[instance["configuration"]] += 1
        total_teu = max(1.0, sum(float(batch.quantity) for batch in batches))
        horizon = max(1.0, max(batch.LT for batch in batches) - min(batch.ET for batch in batches))
        for row in json.loads(path.read_text()):
            values = row["optimisation_objectives"]
            points.append([values["cost"] / total_teu,
                           values["emission"] / total_teu,
                           values["makespan"] / horizon])
        source_files.append({"path": str(path),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not points:
        raise ValueError("no training-only Rule front points found")
    if len(seen_instances) != args.expected_instances:
        raise ValueError(
            f"expected {args.expected_instances} unique training instances, "
            f"found {len(seen_instances)}")
    if args.expected_instances == 450 and configurations != Counter({f"S{i}": 50 for i in range(9)}):
        raise ValueError(f"formal normalization requires 50 instances per S0-S8: {configurations}")
    values = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite normalization source point")
    lower = np.percentile(values, args.lower_percentile, axis=0)
    upper = np.percentile(values, args.upper_percentile, axis=0)
    span = upper - lower
    pad = args.pad_fraction * np.where(span > 0, span, np.maximum(np.abs(lower), 1.0))
    result = {
        "schema_version": 1, "source": "training-only",
        "source_method": "Rule-CCP100",
        "definition": "cost/total_teu, emission/total_teu, makespan/instance_horizon",
        "minimum": (lower - pad).tolist(), "maximum": (upper + pad).tolist(),
        "reference": [1.1, 1.1, 1.1],
        "percentiles": [args.lower_percentile, args.upper_percentile],
        "pad_fraction": args.pad_fraction, "point_count": len(points),
        "instance_count": len(seen_instances),
        "configuration_counts": dict(sorted(configurations.items())),
        "source_files": source_files,
    }
    result["normalization_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print({key: result[key] for key in
           ("point_count", "minimum", "maximum", "normalization_digest")})


if __name__ == "__main__":
    main()
