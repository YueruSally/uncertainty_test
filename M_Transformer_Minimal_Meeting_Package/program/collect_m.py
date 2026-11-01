#!/usr/bin/env python3
"""Collect late-Replace Transformer states for Balanced-teacher imitation."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _state_count(run):
    path = Path(run) / "mutation_policy_states.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip())


def _valid(run, seed, minimum_generation):
    run = Path(run)
    if not (run / "COMPLETE.json").is_file():
        return False
    try:
        complete = _read(run / "COMPLETE.json")
        config = _read(run / "configuration.json")
        control = config["learning_control"]
        return bool(
            complete.get("completed") is True
            and complete.get("component") == "M"
            and config.get("seeds", {}).get("algorithm") == seed
            and control.get("m_forced_operator") == "replace"
            and control.get("m_forced_operator_min_generation")
            == minimum_generation
            and control.get("m_forced_replace_selector") == "model"
            and control.get("model_choice_probabilities", {}).get("M") == 1.0
            and _state_count(run) > 0)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _run(job):
    run, log, command, seed, minimum_generation = job
    if _valid(run, seed, minimum_generation):
        return str(run), "existing", _state_count(run)
    if run.exists():
        raise RuntimeError(f"incomplete output exists; use a fresh root: {run}")
    staging = run.with_name(f".{run.name}.partial.{os.getpid()}")
    if staging.exists():
        raise RuntimeError(f"staging output already exists: {staging}")
    run.parent.mkdir(parents=True, exist_ok=True)
    staged_command = list(command)
    staged_command[staged_command.index("--out") + 1] = str(staging)
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            staged_command, cwd=Path(__file__).resolve().parent,
            stdout=handle, stderr=subprocess.STDOUT,
            env=environment, check=False)
    if result.returncode != 0 or not _valid(
            staging, seed, minimum_generation):
        raise RuntimeError(f"rollout failed with exit {result.returncode}: {log}")
    staging.rename(run)
    return str(run), "complete", _state_count(run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instance-ids", nargs="+", required=True)
    parser.add_argument("--algorithm-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--minimum-generation", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=7)
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
            (args.checkpoint, "checkpoint"), (args.runner, "runner")):
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
    if args.minimum_generation < 0 or args.max_workers < 1:
        parser.error("minimum generation must be nonnegative and workers positive")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for requested in args.instance_ids:
        instance_path = args.instances / f"{requested}.json"
        if not instance_path.is_file():
            parser.error(f"missing instance: {instance_path}")
        instance = requested.replace("train-", "").replace("-", "_")
        for seed in args.algorithm_seeds:
            run = args.out / instance / f"seed_{seed}" / "BOOTSTRAP_TRANSFORMER"
            log = args.out / "logs" / f"{instance}_seed_{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", str(args.runner),
                "--instance", str(instance_path),
                "--normalization", str(args.normalization),
                "--out", str(run),
                "--component", "M",
                "--checkpoint", str(args.checkpoint),
                "--algorithm-seed", str(seed),
                "--inference-action-mode", "argmax",
                "--inference-policy-seed", "61001",
                "--model-choice-probability", "1",
                "--m-forced-operator", "replace",
                "--m-forced-operator-min-generation",
                str(args.minimum_generation),
                "--m-forced-operator-probability", "1",
                "--m-forced-replace-selector", "model",
                "--evaluation-mode", "ccp",
                "--uncertainty-profile", "transport_border_saa_v1",
                "--scenario-count", str(args.scenario_count),
                "--ccp-confidence", str(args.ccp_confidence),
                "--evaluation-budget", str(args.evaluation_budget),
                "--pop", str(args.pop), "--gens", str(args.gens),
                "--device", args.device,
            ]
            jobs.append((run, log, command, seed, args.minimum_generation))

    statuses = Counter()
    states = 0
    print(f"[M-TEACHER-DATA] jobs={len(jobs)} workers={args.max_workers}",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            run, status, count = future.result()
            statuses[status] += 1
            states += count
            print(f"[M-TEACHER-DATA] {index}/{len(jobs)} {status}: "
                  f"{run} states={count}", flush=True)
    result = {
        "completed": True,
        "definition": (
            "late-stage forced-Replace states generated by an initial "
            "path-v4 Transformer; Balanced labels are derived offline"),
        "run_count": len(jobs),
        "instance_count": len(args.instance_ids),
        "algorithm_seed_count": len(args.algorithm_seeds),
        "minimum_generation": args.minimum_generation,
        "policy_state_count": states,
        "status_counts": dict(statuses),
    }
    (args.out / "DATA_COMPLETE.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
