#!/usr/bin/env python3
"""Matched CCP pilot: Base vs Balanced teacher vs distilled M Transformer."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


EPSILON = 1e-12
STRATEGIES = ("BASE", "TEACHER_BALANCED", "TRANSFORMER_BC")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid(run, strategy, seed, minimum_generation):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        control = config["learning_control"]
        common = bool(
            complete.get("completed") is True
            and config.get("seeds", {}).get("algorithm") == seed)
        if strategy == "BASE":
            return common and complete.get("component") == "BASE"
        if strategy == "TEACHER_BALANCED":
            return bool(
                common and complete.get("component") == "M"
                and control.get("m_forced_operator") == "replace"
                and control.get("m_forced_operator_min_generation")
                == minimum_generation
                and control.get("m_forced_replace_selector")
                == "joint_balanced")
        return bool(
            common and complete.get("component") == "M"
            and control.get("m_forced_operator") == "replace"
            and control.get("m_forced_operator_min_generation")
            == minimum_generation
            and control.get("m_forced_replace_selector") == "model"
            and control.get("model_choice_probabilities", {}).get("M") == 1.0)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run(job):
    run, log, command, strategy, seed, minimum_generation = job
    if _valid(run, strategy, seed, minimum_generation):
        return str(run), "existing"
    if run.exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
    run.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(
            run, strategy, seed, minimum_generation):
        raise RuntimeError(f"pilot run failed with exit {result.returncode}: {log}")
    return str(run), "complete"


def _command(args, instance, run, strategy, seed):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", "BASE" if strategy == "BASE" else "M",
        "--algorithm-seed", str(seed),
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", "transport_border_saa_v1",
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop), "--gens", str(args.gens),
        "--device", args.device,
    ]
    if strategy == "TEACHER_BALANCED":
        command.extend([
            "--checkpoint", str(args.bootstrap_checkpoint),
            "--inference-action-mode", "argmax",
            "--model-choice-probability", "0",
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation", str(args.minimum_generation),
            "--m-forced-operator-probability", "1",
            "--m-forced-replace-selector", "joint_balanced",
        ])
    elif strategy == "TRANSFORMER_BC":
        command.extend([
            "--checkpoint", str(args.transformer_checkpoint),
            "--inference-action-mode", "argmax",
            "--inference-policy-seed", "63001",
            "--model-choice-probability", "1",
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation", str(args.minimum_generation),
            "--m-forced-operator-probability", "1",
            "--m-forced-replace-selector", "model",
        ])
    return command


def _strategy_summary(name, rows):
    values = [row["paired_percent"] for row in rows]
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) >= 2 else 0.0
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "strategy": name,
        "runs": len(rows),
        "mean_paired_percent": mean,
        "median_paired_percent": statistics.median(values),
        "standard_deviation_paired_percent": standard_deviation,
        "mean_95ci_low": mean - 1.96 * standard_error,
        "mean_95ci_high": mean + 1.96 * standard_error,
        "minimum_paired_percent": min(values),
        "maximum_paired_percent": max(values),
        "wins": sum(value > EPSILON for value in values),
        "ties": sum(abs(value) <= EPSILON for value in values),
        "losses": sum(value < -EPSILON for value in values),
        "runs_above_5_percent": sum(value >= 5.0 for value in values),
        "runs_above_10_percent": sum(value >= 10.0 for value in values),
        "mean_optimisation_seconds": statistics.fmean(
            row["optimisation_seconds"] for row in rows),
        "mean_model_control_events": statistics.fmean(
            row["model_control_events"] for row in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--minimum-generation", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--evaluation-budget", type=int, default=400)
    parser.add_argument("--scenario-count", type=int, default=200)
    parser.add_argument("--ccp-confidence", type=float, default=0.90)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--runner", type=Path,
        default=Path(__file__).resolve().with_name(
            "run_learning_components_deterministic.py"))
    args = parser.parse_args(argv)
    for path, name in (
            (args.normalization, "normalization"),
            (args.bootstrap_checkpoint, "bootstrap checkpoint"),
            (args.transformer_checkpoint, "Transformer checkpoint"),
            (args.training_report, "training report"),
            (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    normalized_instances = {}
    for requested in args.instance_ids:
        instance_path = args.instances / f"{requested}.json"
        if not instance_path.is_file():
            parser.error(f"missing instance: {instance_path}")
        instance = requested.replace("train-", "").replace("-", "_")
        normalized_instances[instance] = instance_path
        for seed in args.algorithm_seeds:
            for strategy in STRATEGIES:
                run = args.out / instance / f"seed_{seed}" / strategy
                log = args.out / "logs" / f"{instance}_seed_{seed}_{strategy}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((
                    run, log,
                    _command(args, instance_path, run, strategy, seed),
                    strategy, seed, args.minimum_generation))
    statuses = Counter()
    print(f"[M-TRANSFORMER-PILOT] jobs={len(jobs)} workers={args.max_workers}",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            statuses[status] += 1
            print(f"[M-TRANSFORMER-PILOT] {index}/{len(jobs)} "
                  f"{status}: {run}", flush=True)

    rows = []
    scenario_failures = []
    for instance in normalized_instances:
        for seed in args.algorithm_seeds:
            base_run = args.out / instance / f"seed_{seed}" / "BASE"
            base = _read(base_run / "COMPLETE.json")
            base_config = _read(base_run / "configuration.json")
            base_hv = float(base["final_hypervolume"])
            if base_hv <= 0.0:
                raise ValueError(f"nonpositive Base HV: {base_run}")
            for strategy in STRATEGIES:
                run = args.out / instance / f"seed_{seed}" / strategy
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_config["scenario_digest"]:
                    scenario_failures.append(f"{instance}:seed_{seed}:{strategy}")
                hv = float(complete["final_hypervolume"])
                rows.append({
                    "instance": instance,
                    "algorithm_seed": seed,
                    "strategy": strategy,
                    "base_hv": base_hv,
                    "final_hv": hv,
                    "paired_percent": 100.0 * (hv - base_hv) / base_hv,
                    "final_front_size": int(complete["final_front_size"]),
                    "model_control_events": int(
                        complete.get("model_control_events", 0)),
                    "base_control_events": int(
                        complete.get("base_control_events", 0)),
                    "inference_seconds": float(
                        complete.get("inference_seconds", 0.0)),
                    "optimisation_seconds": float(
                        complete["optimisation_seconds"]),
                })
    strategies = [
        _strategy_summary(
            strategy,
            [row for row in rows if row["strategy"] == strategy])
        for strategy in STRATEGIES
    ]
    by_name = {row["strategy"]: row for row in strategies}
    training = _read(args.training_report)
    transformer = by_name["TRANSFORMER_BC"]
    result = {
        "completed": True,
        "definition": (
            "matched frozen-CCP pilot: exact Base, deterministic Balanced "
            "teacher, and a path-v4 Transformer distilled from that teacher"),
        "instance_count": len(normalized_instances),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(rows),
        "status_counts": dict(statuses),
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "teacher_imitation": {
            "training_top_1": training["training"]["top_1_accuracy"],
            "validation_top_1": training["validation"]["top_1_accuracy"],
            "selected_epoch": training["selected_epoch"],
            "training_states": training["training"]["states"],
            "validation_states": training["validation"]["states"],
        },
        "strategies": strategies,
        "pilot_thresholds": {
            "minimum_validation_top_1": 0.50,
            "minimum_transformer_mean_paired_percent": 0.0,
            "minimum_transformer_wins": math.ceil(
                0.5 * len(normalized_instances) * len(args.algorithm_seeds)),
        },
        "runs": rows,
    }
    thresholds = result["pilot_thresholds"]
    result["pilot_pass"] = bool(
        result["scenario_exact"]
        and result["teacher_imitation"]["validation_top_1"]
        >= thresholds["minimum_validation_top_1"]
        and transformer["mean_paired_percent"]
        > thresholds["minimum_transformer_mean_paired_percent"]
        and transformer["wins"] >= thresholds["minimum_transformer_wins"])
    (args.out / "COMPLETE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "run_count", "scenario_exact", "teacher_imitation", "strategies",
        "pilot_thresholds", "pilot_pass")}
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
