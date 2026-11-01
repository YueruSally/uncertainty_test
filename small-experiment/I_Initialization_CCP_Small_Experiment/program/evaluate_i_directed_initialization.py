#!/usr/bin/env python3
"""Screen directed, diverse Rule-I initializers against matched CCP Base."""
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
STRATEGY_PROFILES = {
    "BASE": None,
    "I_DIRECTED_BALANCED": "joint_balanced",
    "I_DIRECTED_BALANCED_P025": "joint_balanced_p025",
    "I_DIRECTED_BALANCED_P050": "joint_balanced_p050",
    "I_DIRECTED_BALANCED_P075": "joint_balanced_p075",
    "I_DIRECTED_BALANCED_HV_GATE": "joint_balanced_hv_gate",
    "I_DIRECTED_DIVERSE": "diverse_cycle",
    "I_DIRECTED_RELIABILITY": "time_reliability",
}
DEFAULT_STRATEGIES = (
    "BASE",
    "I_DIRECTED_BALANCED",
    "I_DIRECTED_DIVERSE",
    "I_DIRECTED_RELIABILITY",
)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def _valid(run, seed, profile, args):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        component = "BASE" if profile is None else "RULE-I"
        return bool(
            complete.get("completed") is True
            and config.get("component") == component
            and config.get("seeds", {}).get("algorithm") == seed
            and config.get("evaluation_mode") == "ccp"
            and config.get("training_size") == args.scenario_count
            and abs(float(config.get("alpha", -1.0))
                    - args.ccp_confidence) <= EPSILON
            and complete.get("real_objective_evaluation_count")
            == args.evaluation_budget
            and (
                profile is None
                or config.get("initialization_rule_profile") == profile))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_path(args, instance, seed, strategy):
    profile = STRATEGY_PROFILES[strategy]
    if profile is None and args.reuse_base_root is not None:
        reused = args.reuse_base_root / instance / f"seed_{seed}" / "BASE"
        if _valid(reused, seed, None, args):
            return reused
    return args.out / instance / f"seed_{seed}" / strategy


def _command(args, instance_path, run, seed, profile):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance_path),
        "--normalization", str(args.normalization),
        "--out", str(run),
        "--component", "BASE" if profile is None else "RULE-I",
        "--algorithm-seed", str(seed),
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", "transport_border_saa_v1",
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if profile is not None:
        command.extend(["--initialization-rule-profile", profile])
    return command


def _execute(job):
    args, instance, instance_path, seed, strategy = job
    profile = STRATEGY_PROFILES[strategy]
    run = _run_path(args, instance, seed, strategy)
    if _valid(run, seed, profile, args):
        status = (
            "reused" if profile is None
            and args.reuse_base_root is not None
            and args.reuse_base_root in run.parents
            else "existing")
        return instance, seed, strategy, run, status
    if run.exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
    run.parent.mkdir(parents=True, exist_ok=True)
    log = run.parent / f"{strategy}.log"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            _command(args, instance_path, run, seed, profile),
            cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(run, seed, profile, args):
        raise RuntimeError(
            f"{instance}:seed_{seed}:{strategy} failed; inspect {log}")
    return instance, seed, strategy, run, "complete"


def _aggregate(args, instance_names, statuses):
    runs = []
    scenario_failures = []
    for instance in instance_names:
        for seed in args.algorithm_seeds:
            base_run = _run_path(args, instance, seed, "BASE")
            base_complete = _read(base_run / "COMPLETE.json")
            base_config = _read(base_run / "configuration.json")
            base_hv = float(base_complete["final_hypervolume"])
            if base_hv <= 0.0:
                raise ValueError(
                    f"nonpositive Base HV: {instance}:seed_{seed}")
            for strategy in args.strategies:
                profile = STRATEGY_PROFILES[strategy]
                run = _run_path(args, instance, seed, strategy)
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_config["scenario_digest"]:
                    scenario_failures.append(
                        f"{instance}:seed_{seed}:{strategy}")
                hv = float(complete["final_hypervolume"])
                runs.append({
                    "instance": instance,
                    "algorithm_seed": seed,
                    "strategy": strategy,
                    "profile": profile,
                    "base_hv": base_hv,
                    "final_hv": hv,
                    "paired_percent": 100.0 * (hv - base_hv) / base_hv,
                    "final_front_size": int(complete["final_front_size"]),
                    "initialization_attempts": int(
                        complete.get("initialization_attempts", 0)),
                    "initializations_different_from_shadow_base": int(
                        complete.get(
                            "initializations_different_from_shadow_base", 0)),
                    "effective_profile_counts": complete.get(
                        "initialization_effective_profile_counts", {}),
                    "initialization_hv_gate_selected": complete.get(
                        "initialization_hv_gate_selected"),
                    "initialization_hv_gate_base_hv": complete.get(
                        "initialization_hv_gate_base_hv"),
                    "initialization_hv_gate_directed_hv": complete.get(
                        "initialization_hv_gate_directed_hv"),
                    "counterfactual_initialization_evaluation_count": int(
                        complete.get(
                            "counterfactual_initialization_evaluation_count",
                            0)),
                    "run_path": str(run),
                })

    strategies = []
    for strategy in args.strategies:
        rows = [row for row in runs if row["strategy"] == strategy]
        values = [row["paired_percent"] for row in rows]
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) >= 2 else 0.0
        se = sd / math.sqrt(len(values))
        strategies.append({
            "strategy": strategy,
            "profile": STRATEGY_PROFILES[strategy],
            "runs": len(rows),
            "mean_paired_percent": mean,
            "standard_deviation_paired_percent": sd,
            "mean_95ci_low": mean - 1.96 * se,
            "mean_95ci_high": mean + 1.96 * se,
            "median_paired_percent": statistics.median(values),
            "minimum_paired_percent": min(values),
            "maximum_paired_percent": max(values),
            "wins": sum(value > EPSILON for value in values),
            "ties": sum(abs(value) <= EPSILON for value in values),
            "losses": sum(value < -EPSILON for value in values),
            "runs_above_5_percent": sum(value >= 5.0 for value in values),
            "runs_above_10_percent": sum(value >= 10.0 for value in values),
            "mean_initialization_attempts": statistics.fmean(
                row["initialization_attempts"] for row in rows),
            "gate_selected_directed": sum(
                row["initialization_hv_gate_selected"] == "directed"
                for row in rows),
            "gate_selected_base": sum(
                row["initialization_hv_gate_selected"] == "base"
                for row in rows),
            "mean_counterfactual_initialization_evaluations": (
                statistics.fmean(
                    row["counterfactual_initialization_evaluation_count"]
                    for row in rows)),
        })

    per_instance = []
    for instance in instance_names:
        instance_result = {"instance": instance, "strategies": []}
        for strategy in args.strategies:
            values = [
                row["paired_percent"] for row in runs
                if row["instance"] == instance
                and row["strategy"] == strategy]
            instance_result["strategies"].append({
                "strategy": strategy,
                "paired_percent_each": values,
                "mean_paired_percent": statistics.fmean(values),
                "minimum_paired_percent": min(values),
                "maximum_paired_percent": max(values),
                "wins": sum(value > EPSILON for value in values),
                "losses": sum(value < -EPSILON for value in values),
            })
        per_instance.append(instance_result)

    alternatives = [
        row for row in strategies if row["strategy"] != "BASE"]
    best = max(alternatives, key=lambda row: row["mean_paired_percent"])
    thresholds = {
        "minimum_mean_paired_percent": 5.0,
        "minimum_win_fraction": 0.70,
        "minimum_paired_percent": -5.0,
    }
    return {
        "completed": True,
        "definition": (
            "matched frozen-CCP screen of directed initialization teachers; "
            "Base path counts, allocation shares, and random stream are paired"),
        "instance_count": len(instance_names),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(runs),
        "status_counts": dict(statuses),
        "reuse_base_root": (
            None if args.reuse_base_root is None
            else str(args.reuse_base_root)),
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "strategies": strategies,
        "best_directed_strategy": best,
        "per_instance": per_instance,
        "screening_thresholds": thresholds,
        "screening_pass": bool(
            not scenario_failures
            and best["mean_paired_percent"] >=
            thresholds["minimum_mean_paired_percent"]
            and best["wins"] / best["runs"] >=
            thresholds["minimum_win_fraction"]
            and best["minimum_paired_percent"] >=
            thresholds["minimum_paired_percent"]),
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-base-root", type=Path)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument(
        "--strategies", nargs="+", choices=tuple(STRATEGY_PROFILES),
        default=list(DEFAULT_STRATEGIES),
        help=("Strategies to evaluate. BASE is required so every directed "
              "run has a matched comparator."))
    parser.add_argument("--max-workers", type=int, default=12)
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
            (args.instances, "instances"),
            (args.normalization, "normalization"),
            (args.runner, "runner")):
        if not path.exists():
            parser.error(f"missing {name}: {path}")
    if args.max_workers < 1:
        parser.error("max-workers must be positive")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("strategies must be unique")
    if "BASE" not in args.strategies:
        parser.error("strategies must include BASE")
    if len(args.strategies) < 2:
        parser.error("at least one directed strategy is required")
    args.out.mkdir(parents=True, exist_ok=True)

    instance_paths = {}
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"missing instance: {path}")
        name = instance_id.replace("train-", "").replace("-", "_")
        instance_paths[name] = path

    jobs = [
        (args, instance, path, seed, strategy)
        for instance, path in sorted(instance_paths.items())
        for seed in args.algorithm_seeds
        for strategy in args.strategies
    ]
    statuses = Counter()
    print(
        f"[I-SCREEN] jobs={len(jobs)} workers={args.max_workers}",
        flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_execute, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            instance, seed, strategy, run, status = future.result()
            statuses[status] += 1
            print(
                f"[I-SCREEN] {index}/{len(jobs)} {status}: "
                f"{instance}:seed_{seed}:{strategy} -> {run}",
                flush=True)

    result = _aggregate(args, sorted(instance_paths), statuses)
    _write(args.out / "COMPLETE.json", result)
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count",
        "algorithm_seed_count", "run_count", "status_counts",
        "scenario_exact", "strategies", "best_directed_strategy",
        "screening_thresholds", "screening_pass")}
    _write(args.out / "QUICK_RESULT.json", quick)
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
