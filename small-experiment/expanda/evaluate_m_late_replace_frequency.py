#!/usr/bin/env python3
"""Compare late-stage replace schedules with matched CCP Base runs."""
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
STRATEGIES = {
    "BASE": (None, False, False, 1, 0),
    "REPLACE_LATE_P050": (0.50, False, False, 1, 0),
    "REPLACE_LATE_P100": (1.00, False, False, 1, 0),
    "REPLACE_LATE_POSITIVE_PROBE": (1.00, True, False, 1, 0),
}
REVERSIBLE_STRATEGY = {
    "REPLACE_LATE_REVERSIBLE_PROBE": (1.00, True, True, 1, 0),
}
RETRY_STRATEGIES = {
    "REPLACE_LATE_REVERSIBLE_RETRY_E05": (1.00, True, True, 2, 5),
    "REPLACE_LATE_REVERSIBLE_RETRY_E10": (1.00, True, True, 2, 10),
}
SCHEDULE_GRID_STRATEGIES = {
    "REPLACE_G04_P100": (1.00, False, False, 1, 0),
    "REPLACE_G06_P100": (1.00, False, False, 1, 0),
    "REPLACE_G08_P100": (1.00, False, False, 1, 0),
    "REPLACE_G10_P050": (0.50, False, False, 1, 0),
    "REPLACE_G10_P100": (1.00, False, False, 1, 0),
    "REPLACE_G12_P050": (0.50, False, False, 1, 0),
    "REPLACE_G12_P100": (1.00, False, False, 1, 0),
    "REPLACE_G15_P050": (0.50, False, False, 1, 0),
    "REPLACE_G15_P100": (1.00, False, False, 1, 0),
}
BEST_OF_STRATEGIES = {
    "REPLACE_G10_BEST3": (1.00, False, False, 1, 0),
    "REPLACE_G10_BEST5": (1.00, False, False, 1, 0),
}
DIRECTED_REPLACE_STRATEGIES = {
    "REPLACE_G10_JOINT_EXTREME": (1.00, False, False, 1, 0),
    "REPLACE_G10_JOINT_BALANCED": (1.00, False, False, 1, 0),
    "REPLACE_G10_JOINT_PARETO": (1.00, False, False, 1, 0),
}
STRATEGY_BEST_OF = {
    "REPLACE_G10_BEST3": 3,
    "REPLACE_G10_BEST5": 5,
}
STRATEGY_REPLACE_SELECTOR = {
    "REPLACE_G10_JOINT_EXTREME": "joint_extreme_cycle",
    "REPLACE_G10_JOINT_BALANCED": "joint_balanced",
    "REPLACE_G10_JOINT_PARETO": "joint_pareto_cycle",
}
STRATEGY_MINIMUM_GENERATIONS = {
    "REPLACE_G04_P100": 4,
    "REPLACE_G06_P100": 6,
    "REPLACE_G08_P100": 8,
    "REPLACE_G10_P050": 10,
    "REPLACE_G10_P100": 10,
    "REPLACE_G12_P050": 12,
    "REPLACE_G12_P100": 12,
    "REPLACE_G15_P050": 15,
    "REPLACE_G15_P100": 15,
    "REPLACE_G10_BEST3": 10,
    "REPLACE_G10_BEST5": 10,
    "REPLACE_G10_JOINT_EXTREME": 10,
    "REPLACE_G10_JOINT_BALANCED": 10,
    "REPLACE_G10_JOINT_PARETO": 10,
}
REUSE_STRATEGY_ALIASES = {
    "REPLACE_G15_P050": "REPLACE_LATE_P050",
    "REPLACE_G15_P100": "REPLACE_LATE_P100",
}
ALL_STRATEGIES = {
    **STRATEGIES,
    **REVERSIBLE_STRATEGY,
    **RETRY_STRATEGIES,
    **SCHEDULE_GRID_STRATEGIES,
    **BEST_OF_STRATEGIES,
    **DIRECTED_REPLACE_STRATEGIES,
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _command(args, instance, out, seed, probability, positive_probe,
             reversible_probe, probe_max_attempts, probe_retry_events,
             minimum_generation, best_of=1,
             replace_selector="base_random"):
    command = [
        sys.executable, "-u", str(args.runner),
        "--instance", str(instance),
        "--normalization", str(args.normalization),
        "--out", str(out),
        "--component", "M",
        "--checkpoint", str(args.checkpoint),
        "--algorithm-seed", str(seed),
        "--inference-action-mode", "argmax",
        "--inference-policy-seed", "45000",
        "--model-choice-probability", "0",
        "--evaluation-mode", "ccp",
        "--uncertainty-profile", "transport_border_saa_v1",
        "--scenario-count", str(args.scenario_count),
        "--ccp-confidence", str(args.ccp_confidence),
        "--evaluation-budget", str(args.evaluation_budget),
        "--pop", str(args.pop),
        "--gens", str(args.gens),
        "--device", args.device,
    ]
    if probability is not None:
        command.extend([
            "--m-forced-operator", "replace",
            "--m-forced-operator-min-generation",
            str(minimum_generation),
            "--m-forced-operator-probability", str(probability),
        ])
        if positive_probe:
            command.append("--m-forced-operator-positive-probe")
        if reversible_probe:
            command.append("--m-forced-operator-reversible-probe")
        if probe_max_attempts != 1:
            command.extend([
                "--m-forced-operator-probe-max-attempts",
                str(probe_max_attempts),
                "--m-forced-operator-probe-retry-events",
                str(probe_retry_events),
            ])
        if best_of != 1:
            command.extend([
                "--m-forced-operator-best-of", str(best_of),
            ])
        if replace_selector != "base_random":
            command.extend([
                "--m-forced-replace-selector", replace_selector,
            ])
    return command


def _valid(run, seed, probability, positive_probe, reversible_probe,
           probe_max_attempts, probe_retry_events, minimum_generation,
           best_of=1, replace_selector="base_random"):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        control = config["learning_control"]
        valid = (
            complete.get("completed") is True
            and complete.get("component") == "M"
            and config.get("component") == "M"
            and config.get("seeds", {}).get("algorithm") == seed
        )
        if probability is None:
            return bool(
                valid
                and control.get("m_forced_operator") is None
                and int(control.get("m_forced_operator_best_of", 1)) == 1
                and control.get(
                    "m_forced_replace_selector", "base_random")
                == "base_random"
                and control.get("model_choice_probabilities", {}).get("M") == 0)
        return bool(
            valid
            and control.get("m_forced_operator") == "replace"
            and control.get("m_forced_operator_min_generation")
            == minimum_generation
            and abs(float(control.get("m_forced_operator_probability"))
                    - probability) <= EPSILON
            and bool(control.get("m_forced_operator_positive_probe"))
            == positive_probe
            and bool(control.get("m_forced_operator_reversible_probe"))
            == reversible_probe
            and int(control.get(
                "m_forced_operator_probe_max_attempts", 1))
            == probe_max_attempts
            and int(control.get(
                "m_forced_operator_probe_retry_events", 0))
            == probe_retry_events
            and int(control.get("m_forced_operator_best_of", 1)) == best_of
            and control.get(
                "m_forced_replace_selector", "base_random")
            == replace_selector
            and (not positive_probe
                 or complete.get("forced_operator_probe_completed") is True))
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run(job):
    (run, log, command, seed, probability, positive_probe, reversible_probe,
     probe_max_attempts, probe_retry_events, minimum_generation, best_of,
     replace_selector) = job
    if _valid(
            run, seed, probability, positive_probe, reversible_probe,
            probe_max_attempts, probe_retry_events, minimum_generation,
            best_of, replace_selector):
        return run, "existing"
    if Path(run).exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
    Path(run).parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with Path(log).open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(
            run, seed, probability, positive_probe, reversible_probe,
            probe_max_attempts, probe_retry_events, minimum_generation,
            best_of, replace_selector):
        raise RuntimeError(f"run failed with exit {result.returncode}: {log}")
    return run, "complete"


def _strategy_run(args, instance, seed, strategy):
    reusable = {
        "BASE", "REPLACE_LATE_P050", "REPLACE_LATE_P100",
        *SCHEDULE_GRID_STRATEGIES,
        *DIRECTED_REPLACE_STRATEGIES,
    }
    if strategy in reusable:
        roots = []
        reuse_root = getattr(args, "reuse_root", None)
        if reuse_root is not None:
            roots.append(reuse_root)
        roots.extend(getattr(args, "reuse_roots", None) or [])
        reused_strategy = REUSE_STRATEGY_ALIASES.get(strategy, strategy)
        for root in roots:
            reused = root / instance / f"seed_{seed}" / reused_strategy
            if (reused / "COMPLETE.json").is_file():
                return reused
    if (args.positive_probe_root is not None
            and strategy == "REPLACE_LATE_POSITIVE_PROBE"):
        reused = (
            args.positive_probe_root / instance / f"seed_{seed}" / strategy)
        if (reused / "COMPLETE.json").is_file():
            return reused
    if (args.reversible_probe_root is not None
            and strategy == "REPLACE_LATE_REVERSIBLE_PROBE"):
        reused = (
            args.reversible_probe_root / instance / f"seed_{seed}" / strategy)
        if (reused / "COMPLETE.json").is_file():
            return reused
    return args.out / instance / f"seed_{seed}" / strategy


def _summary(args, status_counts, instance_names, strategy_specs):
    runs = []
    scenario_failures = []
    for instance in instance_names:
        for seed in args.algorithm_seeds:
            base_run = _strategy_run(args, instance, seed, "BASE")
            base_complete = _read(base_run / "COMPLETE.json")
            base_config = _read(base_run / "configuration.json")
            base_hv = float(base_complete["final_hypervolume"])
            for strategy in strategy_specs:
                run = _strategy_run(args, instance, seed, strategy)
                complete = _read(run / "COMPLETE.json")
                config = _read(run / "configuration.json")
                if config["scenario_digest"] != base_config["scenario_digest"]:
                    scenario_failures.append(
                        f"{instance}:seed_{seed}:{strategy}")
                hv = float(complete["final_hypervolume"])
                percent = 100.0 * (hv - base_hv) / base_hv
                runs.append({
                    "instance": instance,
                    "algorithm_seed": seed,
                    "strategy": strategy,
                    "base_hv": base_hv,
                    "final_hv": hv,
                    "paired_percent": percent,
                    "model_control_events": int(
                        complete.get("model_control_events", 0)),
                    "base_control_events": int(
                        complete.get("base_control_events", 0)),
                    "final_front_size": int(complete["final_front_size"]),
                    "operator_probe_positive": complete.get(
                        "forced_operator_probe_positive"),
                    "operator_probe_delta_hv": complete.get(
                        "forced_operator_probe_delta_hv"),
                    "operator_probe_deltas_hv": complete.get(
                        "forced_operator_probe_deltas_hv",
                        ([complete.get("forced_operator_probe_delta_hv")]
                         if complete.get("forced_operator_probe_completed")
                         else [])),
                    "operator_probe_attempts": int(
                        complete.get(
                            "forced_operator_probe_attempts",
                            int(bool(complete.get(
                                "forced_operator_probe_completed"))))),
                    "operator_probe_fallbacks": int(
                        complete.get("forced_operator_probe_fallbacks", 0)),
                    "operator_probe_evaluations": int(
                        complete.get("forced_operator_probe_evaluations", 0)),
                    "forced_operator_best_of": int(
                        complete.get("forced_operator_best_of", 1)),
                    "forced_replace_selector": complete.get(
                        "forced_replace_selector", "base_random"),
                    "forced_operator_candidate_evaluations": int(
                        complete.get(
                            "forced_operator_candidate_evaluations", 0)),
                })
    strategies = []
    for strategy in strategy_specs:
        rows = [row for row in runs if row["strategy"] == strategy]
        values = [row["paired_percent"] for row in rows]
        standard_deviation = (
            statistics.stdev(values) if len(values) >= 2 else 0.0)
        standard_error = standard_deviation / math.sqrt(len(values))
        mean_value = statistics.fmean(values)
        strategies.append({
            "strategy": strategy,
            "runs": len(rows),
            "mean_paired_percent": mean_value,
            "standard_deviation_paired_percent": standard_deviation,
            "standard_error_paired_percent": standard_error,
            "mean_95ci_low": mean_value - 1.96 * standard_error,
            "mean_95ci_high": mean_value + 1.96 * standard_error,
            "median_paired_percent": statistics.median(values),
            "minimum_paired_percent": min(values),
            "maximum_paired_percent": max(values),
            "wins": sum(value > EPSILON for value in values),
            "win_fraction": (
                sum(value > EPSILON for value in values) / len(values)),
            "ties": sum(abs(value) <= EPSILON for value in values),
            "losses": sum(value < -EPSILON for value in values),
            "runs_above_5_percent": sum(value >= 5.0 for value in values),
            "runs_above_10_percent": sum(value >= 10.0 for value in values),
            "mean_forced_events": statistics.fmean(
                row["model_control_events"] for row in rows),
            "positive_probes": sum(
                row["operator_probe_positive"] is True for row in rows),
            "probe_attempts": sum(
                row["operator_probe_attempts"] for row in rows),
            "probe_fallbacks": sum(
                row["operator_probe_fallbacks"] for row in rows),
            "probe_evaluations": sum(
                row["operator_probe_evaluations"] for row in rows),
            "forced_operator_best_of": max(
                row["forced_operator_best_of"] for row in rows),
            "forced_replace_selector": rows[0]["forced_replace_selector"],
            "mean_candidate_evaluations": statistics.fmean(
                row["forced_operator_candidate_evaluations"] for row in rows),
        })
    per_instance = []
    for instance in instance_names:
        instance_rows = [row for row in runs if row["instance"] == instance]
        instance_result = {"instance": instance, "strategies": []}
        for strategy in strategy_specs:
            rows = [
                row for row in instance_rows if row["strategy"] == strategy]
            values = [row["paired_percent"] for row in rows]
            instance_result["strategies"].append({
                "strategy": strategy,
                "runs": len(rows),
                "paired_percent_each": values,
                "mean_paired_percent": statistics.fmean(values),
                "median_paired_percent": statistics.median(values),
                "minimum_paired_percent": min(values),
                "maximum_paired_percent": max(values),
                "wins": sum(value > EPSILON for value in values),
                "ties": sum(abs(value) <= EPSILON for value in values),
                "losses": sum(value < -EPSILON for value in values),
            })
        per_instance.append(instance_result)
    alternatives = [row for row in strategies if row["strategy"] != "BASE"]
    best = max(alternatives, key=lambda row: row["mean_paired_percent"])
    strategy_names = list(strategy_specs)
    schedule_grid = any(
        strategy in SCHEDULE_GRID_STRATEGIES for strategy in strategy_names)
    result = {
        "completed": True,
        "definition": (
            ("matched frozen-CCP replace timing/probability grid: "
             if schedule_grid else
             "matched frozen-CCP comparison from generation "
             f"{args.minimum_generation}: ")
            + ", ".join(strategy_names)),
        "instance_count": len(instance_names),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "run_count": len(runs),
        "status_counts": status_counts,
        "reuse_root": (
            None if args.reuse_root is None else str(args.reuse_root)),
        "reuse_roots": [str(path) for path in (args.reuse_roots or [])],
        "positive_probe_root": (
            None if args.positive_probe_root is None
            else str(args.positive_probe_root)),
        "reversible_probe_root": (
            None if args.reversible_probe_root is None
            else str(args.reversible_probe_root)),
        "scenario_exact": not scenario_failures,
        "scenario_failures": scenario_failures,
        "strategies": strategies,
        "per_instance": per_instance,
        "best_nonbase_strategy": best,
        "screening_thresholds": {
            "minimum_mean_paired_percent": 2.0,
            "minimum_wins": math.ceil(0.60 * len(args.algorithm_seeds)
                                       * len(instance_names)),
            "minimum_paired_percent": -5.0,
            "minimum_runs_above_5_percent": 1,
        },
        "runs": runs,
    }
    thresholds = result["screening_thresholds"]
    result["screening_pass"] = bool(
        result["scenario_exact"]
        and best["mean_paired_percent"] >=
        thresholds["minimum_mean_paired_percent"]
        and best["wins"] >= thresholds["minimum_wins"]
        and best["minimum_paired_percent"] >=
        thresholds["minimum_paired_percent"]
        and best["runs_above_5_percent"] >=
        thresholds["minimum_runs_above_5_percent"])
    by_name = {row["strategy"]: row for row in strategies}
    p100 = by_name.get("REPLACE_LATE_P100")
    reversible = by_name.get("REPLACE_LATE_REVERSIBLE_PROBE")
    if p100 is not None and reversible is not None:
        result["diagnosis"] = {
            "replace_operator_has_population_signal": bool(
                p100["mean_paired_percent"] >= 2.0
                and p100["wins"] >= math.ceil(0.60 * p100["runs"])),
            "reversible_gate_reduces_losses": bool(
                reversible["losses"] < p100["losses"]
                and reversible["minimum_paired_percent"]
                > p100["minimum_paired_percent"]),
            "reversible_gate_preserves_mean_gain": bool(
                reversible["mean_paired_percent"] >= 0.75
                * max(p100["mean_paired_percent"], 0.0)),
            "p100_mean_paired_percent": p100["mean_paired_percent"],
            "reversible_mean_paired_percent": (
                reversible["mean_paired_percent"]),
            "p100_losses": p100["losses"],
            "reversible_losses": reversible["losses"],
            "p100_minimum_paired_percent": p100["minimum_paired_percent"],
            "reversible_minimum_paired_percent": (
                reversible["minimum_paired_percent"]),
        }
    grid_rows = [
        row for row in strategies
        if row["strategy"] in SCHEDULE_GRID_STRATEGIES]
    if grid_rows:
        best_mean = max(grid_rows, key=lambda row: row["mean_paired_percent"])
        best_median = max(
            grid_rows,
            key=lambda row: (
                row["median_paired_percent"], row["mean_paired_percent"]))
        safest = max(
            grid_rows,
            key=lambda row: (
                row["minimum_paired_percent"],
                row["mean_paired_percent"]))
        result["schedule_grid_diagnosis"] = {
            "best_mean_strategy": best_mean["strategy"],
            "best_mean_paired_percent": best_mean["mean_paired_percent"],
            "best_mean_median_paired_percent": (
                best_mean["median_paired_percent"]),
            "best_mean_minimum_paired_percent": (
                best_mean["minimum_paired_percent"]),
            "best_mean_wins": best_mean["wins"],
            "best_mean_losses": best_mean["losses"],
            "best_median_strategy": best_median["strategy"],
            "best_median_paired_percent": (
                best_median["median_paired_percent"]),
            "safest_strategy": safest["strategy"],
            "safest_minimum_paired_percent": (
                safest["minimum_paired_percent"]),
            "any_schedule_mean_above_5_percent": any(
                row["mean_paired_percent"] >= 5.0 for row in grid_rows),
            "any_schedule_mean_above_10_percent": any(
                row["mean_paired_percent"] >= 10.0 for row in grid_rows),
        }
    best_of_rows = [
        row for row in strategies
        if row["strategy"] in BEST_OF_STRATEGIES]
    if best_of_rows:
        best_of = max(
            best_of_rows, key=lambda row: row["mean_paired_percent"])
        fixed = by_name.get("REPLACE_G10_P100")
        result["best_of_diagnosis"] = {
            "best_strategy": best_of["strategy"],
            "best_mean_paired_percent": best_of["mean_paired_percent"],
            "best_median_paired_percent": best_of["median_paired_percent"],
            "best_minimum_paired_percent": best_of[
                "minimum_paired_percent"],
            "best_wins": best_of["wins"],
            "best_losses": best_of["losses"],
            "fixed_g10_mean_paired_percent": (
                None if fixed is None else fixed["mean_paired_percent"]),
            "best_of_minus_fixed_g10_mean_percent": (
                None if fixed is None else
                best_of["mean_paired_percent"]
                - fixed["mean_paired_percent"]),
            "best_of_mean_above_5_percent": bool(
                best_of["mean_paired_percent"] >= 5.0),
            "best_of_mean_above_10_percent": bool(
                best_of["mean_paired_percent"] >= 10.0),
        }
    directed_rows = [
        row for row in strategies
        if row["strategy"] in DIRECTED_REPLACE_STRATEGIES]
    if directed_rows:
        directed = max(
            directed_rows, key=lambda row: row["mean_paired_percent"])
        fixed = by_name.get("REPLACE_G10_P100")
        result["directed_replace_diagnosis"] = {
            "best_strategy": directed["strategy"],
            "best_mean_paired_percent": directed["mean_paired_percent"],
            "best_median_paired_percent": directed[
                "median_paired_percent"],
            "best_minimum_paired_percent": directed[
                "minimum_paired_percent"],
            "best_wins": directed["wins"],
            "best_losses": directed["losses"],
            "fixed_g10_mean_paired_percent": (
                None if fixed is None else fixed["mean_paired_percent"]),
            "directed_minus_fixed_g10_mean_percent": (
                None if fixed is None else
                directed["mean_paired_percent"]
                - fixed["mean_paired_percent"]),
            "directed_mean_above_5_percent": bool(
                directed["mean_paired_percent"] >= 5.0),
            "directed_mean_above_10_percent": bool(
                directed["mean_paired_percent"] >= 10.0),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reuse-root", type=Path,
        help=("Reuse completed BASE/P050/P100 runs from this compatible root; "
              "only the positive-probe strategy is written to --out."))
    parser.add_argument(
        "--reuse-roots", nargs="+", type=Path,
        help=("Search these compatible roots in order for reusable Base and "
              "generation-15 P050/P100 runs; missing runs go to --out."))
    parser.add_argument(
        "--positive-probe-root", type=Path,
        help="Reuse completed positive-probe runs from this compatible root.")
    parser.add_argument(
        "--reversible-probe-root", type=Path,
        help="Reuse completed one-attempt reversible-probe runs.")
    parser.add_argument(
        "--include-reversible-probe", action="store_true",
        help=("Also evaluate the budget-counted reversible positive-probe "
              "strategy."))
    parser.add_argument(
        "--include-reversible-retry-probes", action="store_true",
        help=("Also evaluate two-attempt reversible probes with 5- and "
              "10-event retry intervals."))
    parser.add_argument(
        "--strategies", nargs="+", choices=sorted(ALL_STRATEGIES),
        help=("Explicit strategy subset. When omitted, preserve the legacy "
              "frequency-screen selection controlled by the include flags."))
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int,
                        required=True)
    parser.add_argument("--minimum-generation", type=int, default=15)
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
    if not args.instance_ids or not args.algorithm_seeds:
        parser.error("at least one instance and one algorithm seed are required")
    for path, name in (
            (args.normalization, "normalization"),
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    instance_paths = {}
    for instance_id in args.instance_ids:
        path = args.instances / f"{instance_id}.json"
        if not path.is_file():
            parser.error(f"missing instance: {path}")
        instance_paths[instance_id.replace("train-", "").replace("-", "_")] = path

    if args.strategies:
        if "BASE" not in args.strategies:
            parser.error("explicit --strategies must include BASE")
        if len(set(args.strategies)) != len(args.strategies):
            parser.error("explicit --strategies must not contain duplicates")
        strategy_specs = {
            strategy: ALL_STRATEGIES[strategy]
            for strategy in args.strategies
        }
    else:
        strategy_specs = dict(STRATEGIES)
        if args.include_reversible_probe:
            strategy_specs.update(REVERSIBLE_STRATEGY)
        if args.include_reversible_retry_probes:
            if not args.include_reversible_probe:
                parser.error(
                    "retry probes require --include-reversible-probe")
            strategy_specs.update(RETRY_STRATEGIES)
    jobs = []
    for instance, instance_path in sorted(instance_paths.items()):
        for seed in args.algorithm_seeds:
            for strategy, specification in strategy_specs.items():
                (probability, positive_probe, reversible_probe,
                 probe_max_attempts, probe_retry_events) = specification
                run = _strategy_run(args, instance, seed, strategy)
                minimum_generation = STRATEGY_MINIMUM_GENERATIONS.get(
                    strategy, args.minimum_generation)
                best_of = STRATEGY_BEST_OF.get(strategy, 1)
                replace_selector = STRATEGY_REPLACE_SELECTOR.get(
                    strategy, "base_random")
                jobs.append((
                    run, run.with_suffix(".nohup.log"),
                    _command(
                        args, instance_path, run, seed, probability,
                        positive_probe, reversible_probe,
                        probe_max_attempts, probe_retry_events,
                        minimum_generation, best_of, replace_selector),
                    seed, probability, positive_probe, reversible_probe,
                    probe_max_attempts, probe_retry_events,
                    minimum_generation, best_of, replace_selector))
    counts = Counter()
    print(f"[FREQUENCY] jobs={len(jobs)} workers={args.max_workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_run, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            run, status = future.result()
            counts[status] += 1
            print(
                f"[FREQUENCY] {index}/{len(jobs)} {status}: {run}",
                flush=True)
    result = _summary(
        args, dict(counts), sorted(instance_paths), strategy_specs)
    (args.out / "COMPLETE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    quick = {key: result[key] for key in (
        "completed", "definition", "instance_count", "algorithm_seed_count",
        "run_count", "status_counts", "scenario_exact", "strategies",
        "per_instance", "best_nonbase_strategy", "screening_thresholds",
        "screening_pass")}
    if "diagnosis" in result:
        quick["diagnosis"] = result["diagnosis"]
    if "schedule_grid_diagnosis" in result:
        quick["schedule_grid_diagnosis"] = (
            result["schedule_grid_diagnosis"])
    if "best_of_diagnosis" in result:
        quick["best_of_diagnosis"] = result["best_of_diagnosis"]
    if "directed_replace_diagnosis" in result:
        quick["directed_replace_diagnosis"] = (
            result["directed_replace_diagnosis"])
    (args.out / "QUICK_RESULT.json").write_text(
        json.dumps(quick, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(quick, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
