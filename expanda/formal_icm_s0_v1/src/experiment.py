#!/usr/bin/env python3
"""Common-seed I/C/M single-component and combined frozen K8 screen."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / "support"))
sys.path.insert(0, str(SOURCE))

from crossover import FrozenCResidualGate
from combined import CombinedFrozenK8Controller
from runtime_common import paired_summary
from initialization import FrozenIResidualGate
from runtime_common import digest, read_json, write_json
from mutation import (
    FrozenScaleRanker, ScaleSearchController, ScaleSearchPolicy)
from runtime_common import _normalizer
from transformer_ppo.archive import RewardArchive, RewardConfig
from crossover import CrossoverK8Controller
from transformer_ppo.episode import run_episode
from initialization import InitializationK8Controller
from transformer_ppo.instance_catalog import canonical_digest, read_instance


BASE = "BASE"
I_HEURISTIC = "I_DIRECTED_BALANCED"
I_TRANSFORMER = "I_K8_RESIDUAL"
C_HEURISTIC = "C_K8_SIMPLE"
C_TRANSFORMER = "C_K8_RESIDUAL"
M_HEURISTIC = "M_BALANCED"
M_TRANSFORMER = "M_K8_TRANSFORMER"
ICM_TRANSFORMER = "I_C_M_K8_TRANSFORMER"
METHODS = (BASE, I_HEURISTIC, I_TRANSFORMER, C_HEURISTIC,
           C_TRANSFORMER, M_HEURISTIC, M_TRANSFORMER, ICM_TRANSFORMER)
INSTANCES = tuple(f"prescreen-search-S0-{index:03d}" for index in range(1, 4))
SEEDS = tuple(range(930001, 930011))
SETTINGS = {"population_size": 20, "maximum_generations": 1000, "evaluation_budget": 400, "scenario_count": 100, "ccp_confidence": .9, "per_run_timeout_seconds": 900, "invocation_limit_seconds": 7200}

def configure(config, catalog):
    global SETTINGS, INSTANCES, SEEDS
    SETTINGS = dict(config)
    INSTANCES = tuple(row["instance_id"] for row in catalog["instances"])
    SEEDS = tuple(config["algorithm_seeds"])
    if tuple(config["methods"]) != METHODS:
        raise ValueError("method list changed")



def log(message):
    print(f"[ICM-COMMON] {message}", flush=True)


def run_worker(a):
    torch.set_num_threads(1)
    info, _ = read_instance(a.instance)
    holders = []
    component = "BASE"; trainer = None; factories = None; override = None

    if a.method == BASE:
        pass
    elif a.method in (I_HEURISTIC, I_TRANSFORMER):
        policy = (None if a.method == I_HEURISTIC else
                  FrozenIResidualGate(read_json(a.i_bundle), digest))

        def factory(*args, **kwargs):
            controller = InitializationK8Controller(
                *args,
                selection_mode=("balanced" if policy is None else "transformer"),
                selection_policy=policy, pool_size=8, **kwargs)
            holders.append(controller); return controller
        factories = {"BASE": factory}; override = a.method
    elif a.method in (C_HEURISTIC, C_TRANSFORMER):
        policy = (None if a.method == C_HEURISTIC else
                  FrozenCResidualGate(read_json(a.c_bundle), digest))

        def factory(*args, **kwargs):
            controller = CrossoverK8Controller(
                *args,
                selection_mode=("simple" if policy is None else "transformer"),
                selection_policy=policy, pool_size=8, **kwargs)
            holders.append(controller); return controller
        factories = {"BASE": factory}; override = a.method
    elif a.method in (M_HEURISTIC, M_TRANSFORMER):
        predictor = (None if a.method == M_HEURISTIC else
                     FrozenScaleRanker(read_json(a.m_bundle), digest))
        internal = ("HEURISTIC_BALANCED" if predictor is None
                    else "TRANSFORMER_K8_SIZE200")
        trainer = ScaleSearchPolicy(
            info["instance_id"], a.seed, internal, predictor=predictor)

        def factory(*args, **kwargs):
            controller = ScaleSearchController(*args, **kwargs)
            holders.append(controller); return controller
        factories = {"M": factory}; component = "M"
    elif a.method == ICM_TRANSFORMER:
        predictor = FrozenScaleRanker(read_json(a.m_bundle), digest)
        m_policy = ScaleSearchPolicy(
            info["instance_id"], a.seed, "TRANSFORMER_K8_SIZE200",
            predictor=predictor)

        def factory(*args, **kwargs):
            normalizer = kwargs.pop("normalizer")
            reliable_options = kwargs.pop("reliable_options")
            batches = kwargs.pop("batches")
            arcs = kwargs.pop("arcs")
            controller = CombinedFrozenK8Controller(
                *args, trainer=m_policy, archive=RewardArchive(normalizer),
                reward_config=RewardConfig(), batches=batches,
                generations=SETTINGS["maximum_generations"], pop_size=SETTINGS["population_size"], graph_observation={},
                i_bundle=a.i_bundle, c_bundle=a.c_bundle,
                reliable_options=reliable_options, arcs=arcs, **kwargs)
            holders.append(controller); return controller
        factories = {"BASE": factory}; override = a.method
    else:
        raise ValueError(f"unknown common-seed method {a.method}")

    result = run_episode(
        a.data, a.instance, a.out, trainer, _normalizer(a.normalization),
        algorithm_seed=a.seed, evaluation_budget=SETTINGS["evaluation_budget"], pop_size=SETTINGS["population_size"],
        generations=SETTINGS["maximum_generations"], evaluation_mode="ccp", component=component,
        scenario_count=SETTINGS["scenario_count"], ccp_confidence=SETTINGS["ccp_confidence"],
        uncertainty_profile="transport_border_saa_v1",
        inference_controller_factories=factories,
        method_override=override)
    if a.method != BASE and len(holders) != 1:
        raise RuntimeError("common-seed controller was not installed once")
    result.update({
        "strategy": a.method, "instance": info["instance_id"],
        "algorithm_seed": a.seed, "plan_sha256": a.plan_sha,
        "scenario_digest": read_json(a.out / "configuration.json")[
            "scenario_digest"],
        "new_training_steps": 0, "new_ppo_updates": 0,
        "learning_enabled": False,
        "role": "formal S0 frozen-component selection; not final test",
        "population_size": SETTINGS["population_size"],
        "maximum_generations": SETTINGS["maximum_generations"],
    })
    validate_run(result, a.method, a.seed, a.plan_sha)
    write_json(a.out / "RUN_COMPLETE.json", result)


def validate_run(row, method, seed, plan_sha):
    if (row.get("completed") is not True
            or row.get("strategy") != method
            or row.get("algorithm_seed") != seed
            or row.get("plan_sha256") != plan_sha
            or row.get("real_objective_evaluation_count") != SETTINGS["evaluation_budget"]
            or row.get("evaluation_budget") != SETTINGS["evaluation_budget"]
            or row.get("ppo_updates", 0)
            or not math.isfinite(float(row.get("final_hypervolume", math.nan)))
            or float(row.get("final_hypervolume", 0.0)) < 0.0):
        raise ValueError("invalid common-seed run")
    if method in (I_HEURISTIC, I_TRANSFORMER):
        if (row.get("initialization_k8_hidden_candidate_objective_evaluations") != 0
                or row.get("initialization_k8_pool_events") != SETTINGS["population_size"]):
            raise ValueError("invalid common-seed I run")
    if method in (C_HEURISTIC, C_TRANSFORMER):
        if (row.get("crossover_k8_hidden_candidate_objective_evaluations") != 0
                or row.get("crossover_k8_eligible_pool_events", 0) <= 0):
            raise ValueError("invalid common-seed C run")
    if method in (M_HEURISTIC, M_TRANSFORMER):
        audit = row.get("k8_scale_search", {})
        if (audit.get("hidden_candidate_objective_evaluations") != 0
                or audit.get("pool_events", 0) <= 0):
            raise ValueError("invalid common-seed M run")
    if method == ICM_TRANSFORMER:
        if (row.get("combined_k8_protocol")
                != "frozen-i-c-m-common-seed-v1"
                or row.get("combined_hidden_candidate_objective_evaluations") != 0):
            raise ValueError("invalid common-seed I+C+M run")


def summarize(runs):
    keyed = {(row["instance"], row["algorithm_seed"], row["strategy"]): row
             for row in runs}
    expected = {(name, seed, method) for name in INSTANCES
                for seed in SEEDS for method in METHODS}
    if len(keyed) != len(runs) or set(keyed) != expected:
        raise ValueError("incomplete or duplicate common-seed grid")
    def percent(value, reference):
        return 100.0*(value/reference-1.0) if reference > 1e-15 else None
    paired = []
    for name in INSTANCES:
        for seed in SEEDS:
            base = keyed[name, seed, BASE]
            scenario = base["scenario_digest"]
            pairing = base.get("pairing")
            for method in METHODS:
                row = keyed[name, seed, method]
                if (row["scenario_digest"] != scenario
                        or row.get("pairing") != pairing):
                    raise ValueError("common-seed pairing or scenarios changed")
                record = {
                    "instance": name, "algorithm_seed": seed,
                    "strategy": method,
                    "percent_vs_base": percent(row["final_hypervolume"], base["final_hypervolume"]),
                    "absolute_vs_base": row["final_hypervolume"]-base["final_hypervolume"],
                }
                for reference in (I_HEURISTIC, C_HEURISTIC, M_HEURISTIC,
                                  I_TRANSFORMER, C_TRANSFORMER, M_TRANSFORMER):
                    ref = keyed[name, seed, reference]["final_hypervolume"]
                    record[f"percent_vs_{reference.lower()}"] = percent(row["final_hypervolume"], ref)
                    record[f"absolute_vs_{reference.lower()}"] = row["final_hypervolume"]-ref
                paired.append(record)
    methods = {}; table = []
    for index, method in enumerate(METHODS):
        selected = [row for row in paired if row["strategy"] == method]
        actual = [row for row in runs if row["strategy"] == method]
        methods[method] = {
            "vs_base": paired_summary(selected, "percent_vs_base", 931100+index),
            "absolute_vs_base": paired_summary(selected, "absolute_vs_base", 931100+index),
            "mean_final_hv": statistics.fmean(r["final_hypervolume"] for r in actual),
            "mean_seconds": statistics.fmean(
                row["optimisation_seconds"] for row in actual),
            "logical_objective_evaluations": sum(
                row["real_objective_evaluation_count"] for row in actual),
        }
        table.append({
            "method": method,
            "mean_final_hv": methods[method]["mean_final_hv"],
            "mean_absolute_hv_gain_vs_base": methods[method]["absolute_vs_base"]["mean"],
            "undefined_percent_pairs": methods[method]["vs_base"]["undefined_pairs"],
            "mean_percent_vs_base": methods[method]["vs_base"]["mean"],
            "ci_low": methods[method]["vs_base"][
                "hierarchical_bootstrap_95ci_low"],
            "ci_high": methods[method]["vs_base"][
                "hierarchical_bootstrap_95ci_high"],
            "wins": methods[method]["vs_base"]["wins"],
            "ties": methods[method]["vs_base"]["ties"],
            "losses": methods[method]["vs_base"]["losses"],
            "mean_seconds": methods[method]["mean_seconds"],
        })
    component_pairs = ((I_TRANSFORMER, I_HEURISTIC),
                       (C_TRANSFORMER, C_HEURISTIC),
                       (M_TRANSFORMER, M_HEURISTIC))
    component_comparisons = {}
    for index, (learned, heuristic) in enumerate(component_pairs):
        selected = [row for row in paired if row["strategy"] == learned]
        key = f"percent_vs_{heuristic.lower()}"
        component_comparisons[learned] = {
            "heuristic": heuristic,
            "summary": paired_summary(selected, key, 931500+index),
            "absolute_hv_gain": paired_summary(selected, f"absolute_vs_{heuristic.lower()}", 931500+index),
        }
    combined_rows = [row for row in paired
                     if row["strategy"] == ICM_TRANSFORMER]
    combined_vs_singles = {}
    for index, method in enumerate(
            (I_TRANSFORMER, C_TRANSFORMER, M_TRANSFORMER)):
        combined_vs_singles[method] = paired_summary(
            combined_rows, f"percent_vs_{method.lower()}", 931600+index)
    combined_absolute = {method: paired_summary(combined_rows, f"absolute_vs_{method.lower()}", 931600+i)
        for i, method in enumerate((I_TRANSFORMER, C_TRANSFORMER, M_TRANSFORMER))}
    ranking_basis = "absolute_vs_base" if methods[BASE]["vs_base"]["undefined_pairs"] else "vs_base"
    single_means = {method: methods[method][ranking_basis]["mean"]
                    for method in (I_TRANSFORMER, C_TRANSFORMER, M_TRANSFORMER)}
    best_single = max(single_means, key=single_means.get)
    return {
        "completed": True, "reference_method": BASE,
        "methods": methods, "table": table, "paired_runs": paired,
        "component_comparisons": component_comparisons,
        "best_single_component": best_single,
        "best_single_mean_percent_vs_base": methods[best_single]["vs_base"]["mean"],
        "ranking_basis": ranking_basis,
        "ranking_is_descriptive_not_proof_of_superiority": True,
        "combined_absolute_vs_single_components": combined_absolute,
        "combined_vs_single_components": combined_vs_singles,
        "combined_vs_best_single": combined_vs_singles[best_single],
        "next_action": "review paired component comparisons and independent-scenario re-evaluation; no automatic PPO",
    }


def bundles_compatible(i_bundle, c_bundle, m_bundle):
    return (i_bundle.get("protocol") == "i-k8-directed-residual-v1"
            and c_bundle.get("protocol") == "c-k8-simple-residual-v1"
            and m_bundle.get("protocol")
            == "m-prescreen-scale-learning-curve-v1"
            and len(m_bundle.get("models", {}).get("200", {}).get(
                "checkpoints", ())) == 3)


def frozen_plan(a):
    i_bundle = read_json(a.i_bundle); c_bundle = read_json(a.c_bundle)
    m_bundle = read_json(a.m_bundle)
    if not bundles_compatible(i_bundle, c_bundle, m_bundle):
        raise ValueError("common-seed frozen bundles are incompatible")
    catalog = read_json(a.search_instances / "catalog.json")
    identity = dict(catalog); claimed = identity.pop("catalog_digest", None)
    if (canonical_digest(identity) != claimed or catalog.get("count") != len(INSTANCES)
            or catalog.get("final_test_instances_used") is not False
            or [row["instance_id"] for row in catalog["instances"]]
            != list(INSTANCES)):
        raise ValueError("common-seed instance catalog changed")
    inputs = [a.data.resolve(), a.normalization.resolve(), a.config.resolve(), a.i_bundle.resolve(), a.c_bundle.resolve(), a.m_bundle.resolve(),
              a.search_instances / "catalog.json", SOURCE.parent / "run.py"]
    inputs.extend(sorted(SOURCE.rglob("*.py")))
    inputs.append(SOURCE / "data/border_crossing_events.csv")
    for component, bundle in (("I", i_bundle), ("C", c_bundle), ("M", m_bundle)):
        inputs.append(SOURCE.parent / "assets" / component / "BUNDLE.json")
        spec = bundle["models"]["200"] if component == "M" else bundle
        inputs.extend(Path(path) for path in spec["checkpoints"])
    inputs.extend(a.search_instances / f"{name}.json" for name in INSTANCES)
    return {
        "protocol": SETTINGS["protocol"], "settings": SETTINGS,
        "instances": list(INSTANCES), "algorithm_seeds": list(SEEDS),
        "methods": list(METHODS), "runs": len(INSTANCES)*len(SEEDS)*len(METHODS),
        "population_size": SETTINGS["population_size"],
        "maximum_generations": SETTINGS["maximum_generations"],
        "evaluation_budget": SETTINGS["evaluation_budget"],
        "scenario_count": SETTINGS["scenario_count"],
        "ccp_confidence": SETTINGS["ccp_confidence"],
        "workers": a.workers, "ppo_enabled": False, "opens_locked_final_test": False,
        "input_sha256": {str(path.resolve()): digest(path) for path in inputs},
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
    }



def freeze_or_verify_plan(plan_path, plan):
    if plan_path.exists():
        previous = read_json(plan_path)
        if previous != plan:
            changed = sorted(key for key in previous.keys() | plan.keys()
                             if previous.get(key) != plan.get(key))
            raise ValueError(
                f"frozen common-seed plan changed ({', '.join(changed)}). "
                f"Preserve {plan_path.parent} and restart with a new --out "
                "directory. Do not delete PLAN.json or mix runs from "
                "different code/model versions.")
        return
    write_json(plan_path, plan)


def execute(a):
    root = a.out.resolve(); plan = frozen_plan(a)
    plan_path = root / "PLAN.json"
    freeze_or_verify_plan(plan_path, plan)
    if (root / "SEARCH_COMPLETE").exists():
        log("already complete; see QUICK_RESULT.json"); return
    if a.workers != SETTINGS["workers"]: raise ValueError("Workers differ from the frozen configuration")
    plan_sha = digest(plan_path); (root / "logs").mkdir(exist_ok=True)
    started = time.monotonic(); deadline = started + SETTINGS["invocation_limit_seconds"]
    status = {"stage": "search", "completed_runs": 0,
              "total_runs": plan["runs"]}
    lock = threading.Lock(); stop = threading.Event()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    log(f"{plan['runs']} common-seed searches; no PPO and no hidden evaluation")

    def task(name, seed, method):
        if stop.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("invocation stopped; completed runs remain reusable")
        folder = root / "runs" / name / f"seed_{seed}" / method
        reused = (folder / "RUN_COMPLETE.json").exists()
        if not reused:
            if folder.exists():
                quarantine = root / "interrupted_runs" / (
                    f"{name}_{seed}_{method}_{time.time_ns()}")
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(folder, quarantine)
            logfile = root / "logs" / f"{name}_{seed}_{method}_{time.time_ns()}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(a.config.resolve()),
                "--worker", "--data", str(a.data.resolve()),
                "--search-instances", str(a.search_instances.resolve()),
                "--normalization", str(a.normalization.resolve()),
                "--i-bundle", str(a.i_bundle.resolve()),
                "--c-bundle", str(a.c_bundle.resolve()),
                "--m-bundle", str(a.m_bundle.resolve()),
                "--out", str(folder),
                "--instance", str((a.search_instances / f"{name}.json").resolve()),
                "--seed", str(seed), "--method", method,
                "--plan-sha", plan_sha]
            with logfile.open("w") as handle, subprocess.Popen(
                    command, cwd=SOURCE, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT) as process:
                begin = time.monotonic()
                while process.poll() is None:
                    now = time.monotonic()
                    if stop.is_set() or now >= deadline or now-begin >= SETTINGS["per_run_timeout_seconds"]:
                        process.kill(); process.wait()
                        raise TimeoutError(f"run stopped; see {logfile}")
                    time.sleep(.2)
                if process.returncode:
                    tail = "\n".join(logfile.read_text(
                        errors="replace").splitlines()[-30:])
                    raise RuntimeError(f"worker failed: {logfile}\n{tail}")
        row = read_json(folder / "RUN_COMPLETE.json")
        validate_run(row, method, seed, plan_sha)
        with lock:
            status["completed_runs"] += 1
            status["elapsed_seconds"] = time.monotonic()-started
            write_json(root / "STATUS.json", status)
            log(f"{status['completed_runs']}/{plan['runs']} "
                f"{'reused' if reused else 'complete'} {name}/{seed}/{method} "
                f"HV={row['final_hypervolume']:.7g}")
        return row

    # Complete one combined run before launching independent jobs, so an
    # integration failure cannot interrupt concurrently running comparisons.
    smoke = (INSTANCES[0], SEEDS[0], ICM_TRANSFORMER)
    log("first verify one complete I+C+M run; it counts toward the frozen full grid")
    runs = [task(*smoke)]
    if a.smoke:
        runs.extend(task(INSTANCES[0], SEEDS[0], method)
                    for method in METHODS if method != ICM_TRANSFORMER)
        benchmark = {row["strategy"]: row["optimisation_seconds"] for row in runs}
        estimate = sum(benchmark.values())*len(INSTANCES)*len(SEEDS)
        result = {"completed": True, "role": "runtime and correctness preflight only",
                  "settings": SETTINGS, "methods_checked": list(METHODS),
                  "runs": runs, "seconds_per_method": benchmark,
                  "estimated_search_serial_hours": estimate/3600,
                  "estimated_search_parallel_hours_ideal": estimate/3600/a.workers,
                  "estimate_excludes_oos_and_parallel_contention": True,
                  "not_used_to_select_checkpoints": True}
        write_json(root / "SMOKE_REPORT.json", result)
        write_json(root / "STATUS.json", {**status, "stage": "smoke_complete"})
        log("SMOKE COMPLETE; run --run for the frozen full grid; no model selection")
        return
    method_order = (ICM_TRANSFORMER, *(
        method for method in METHODS if method != ICM_TRANSFORMER))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(task, name, seed, method)
                   for name in INSTANCES for seed in SEEDS
                   for method in method_order
                   if (name, seed, method) != smoke]
        try:
            for future in as_completed(futures):
                runs.append(future.result())
        except BaseException:
            stop.set()
            for future in futures: future.cancel()
            raise
    report = summarize(runs)
    report.update({"design": plan, "completed_searches": len(runs),
        "new_training_steps": 0, "new_ppo_updates": 0,
        "search_objective_evaluations": sum(row["real_objective_evaluation_count"] for row in runs),
        "hidden_candidate_objective_evaluations": 0,
        "elapsed_seconds_this_invocation": time.monotonic()-started,
        "oos_completed": False,
        "interrupted_attempt_cost_not_in_completed_total": (root / "interrupted_runs").exists()})
    write_json(root / "COMPLETE.json", {**report, "runs": runs})
    write_json(root / "QUICK_RESULT.json", {
        key: value for key, value in report.items()
        if key not in ("runs", "paired_runs")})
    with (root / "RESULT_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["table"][0]))
        writer.writeheader(); writer.writerows(report["table"])
    write_json(root / "STATUS.json", {**status, "stage": "search_complete",
        "best_single_component": report["best_single_component"]})
    (root / "SEARCH_COMPLETE").write_text(
        report["best_single_component"] + "\n")
    print((root / "RESULT_TABLE.csv").read_text(), flush=True)
    log(f"COMPLETE best_single={report['best_single_component']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
        default=SOURCE.parent / "assets/data/data_expanded.xlsx")
    for name in ("search-instances", "normalization", "i-bundle",
                 "c-bundle", "m-bundle", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", choices=(1,2,3,4), default=3, type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--plan-sha")
    a = parser.parse_args(); a.out = a.out.resolve()
    configure(read_json(a.config), read_json(a.search_instances / "catalog.json"))
    if a.worker:
        if None in (a.instance, a.seed, a.method, a.plan_sha):
            parser.error("worker arguments incomplete")
        a.out.mkdir(parents=True, exist_ok=False); run_worker(a); return
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(a)
        except Exception as exc:
            previous = (read_json(a.out / "STATUS.json")
                        if (a.out / "STATUS.json").exists() else {})
            write_json(a.out / "STATUS.json", {
                **previous, "stage": "stopped", "error": str(exc)})
            log(f"STOPPED {exc}")
            raise


if __name__ == "__main__":
    main()
