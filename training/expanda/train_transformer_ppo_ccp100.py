#!/usr/bin/env python3
"""Train one separate M-location, C-cut, or I-path PPO seed."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from transformer_ppo.archive import ObjectiveNormalizer, RewardConfig
from transformer_ppo.config import PPOConfig, SCENARIOS, episode_seeds
from transformer_ppo.crossover_observations import CROSSOVER_GRAPH_MODEL_SPEC
from transformer_ppo.graph_features import GRAPH_MODEL_SPEC
from transformer_ppo.initialization_observations import INITIALIZATION_GRAPH_MODEL_SPEC
from transformer_ppo.observations import (
    CANDIDATE_FEATURES, GLOBAL_FEATURES, PARENT_FEATURES,
)
from transformer_ppo.episode import run_episode
from transformer_ppo.ppo import PPOTrainer


def _normalizer(path):
    value = json.loads(path.read_text())
    if value.get("source") != "training-only":
        raise ValueError("normalization must explicitly declare source=training-only")
    return ObjectiveNormalizer(tuple(value["minimum"]), tuple(value["maximum"]),
                               tuple(value.get("reference", (1.1, 1.1, 1.1))))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_manifests(instances, policy_seed, max_instances, configuration=None,
                       instance_ids=None):
    manifests = []
    records = {}
    for path in sorted(instances.glob("train-S*.json")):
        record = _read_json(path)
        if (record.get("split") == "train"
                and (configuration is None
                     or record.get("configuration") == configuration)):
            manifests.append(path)
            records[path.stem] = record
    if not manifests:
        scope = f" for configuration {configuration}" if configuration else ""
        raise ValueError(
            f"no training manifests found{scope} in {instances.resolve()}")
    expected = 50 if configuration is not None else 450
    if (len(manifests) != expected and max_instances is None
            and instance_ids is None):
        raise ValueError(
            f"formal training requires exactly {expected} manifests, "
            f"found {len(manifests)}")
    if instance_ids is not None:
        if max_instances is not None:
            raise ValueError("--instance-ids and --max-instances are mutually exclusive")
        if not instance_ids or len(set(instance_ids)) != len(instance_ids):
            raise ValueError("--instance-ids must be a non-empty list of unique IDs")
        available = {path.stem: path for path in manifests}
        missing = [instance_id for instance_id in instance_ids
                   if instance_id not in available]
        if missing:
            raise ValueError(
                f"requested training manifests are unavailable: {missing}")
        manifests = [available[instance_id] for instance_id in instance_ids]
    else:
        random.Random(policy_seed).shuffle(manifests)
    if max_instances is not None:
        if max_instances < 1:
            raise ValueError("--max-instances must be positive")
        if len(manifests) < max_instances:
            raise ValueError(
                f"--max-instances={max_instances} requested but only "
                f"{len(manifests)} matching manifests were found")
        manifests = manifests[:max_instances]
    return manifests, records


def _catalog_digest(instances):
    catalog_path = instances / "catalog.json"
    if not catalog_path.exists():
        return None
    return _read_json(catalog_path).get("catalog_digest")


def _model_spec(component):
    if component == "C":
        return dict(CROSSOVER_GRAPH_MODEL_SPEC)
    if component == "I":
        return dict(INITIALIZATION_GRAPH_MODEL_SPEC)
    return {
        "parent_dim": len(PARENT_FEATURES),
        "candidate_dim": len(CANDIDATE_FEATURES),
        "global_dim": len(GLOBAL_FEATURES),
        "action_count": 5,
        **GRAPH_MODEL_SPEC,
    }


def _run_config(args, config, manifests, records):
    normalization = _read_json(args.normalization)
    return {
        "schema_version": 3,
        "component": args.component,
        "architecture": args.architecture,
        "model_spec": _model_spec(args.component),
        "network_encoding": "shared-network-tokens-v1",
        "policy_seed": args.policy_seed,
        "ppo": asdict(config),
        "ppo_digest": config.digest(),
        "episode_count": len(manifests),
        "ordered_instances": [{
            "instance_id": path.stem,
            "instance_digest": records[path.stem]["instance_digest"],
        } for path in manifests],
        "catalog_digest": _catalog_digest(args.instances),
        "source_data_sha256": _sha256(args.data),
        "evaluation_budget_per_episode": args.evaluation_budget,
        "evaluation_mode": args.evaluation_mode,
        "configuration": args.configuration,
        "population": args.pop,
        "generations": args.gens,
        "normalization_sha256": _sha256(args.normalization),
        "normalization_digest": normalization.get("normalization_digest"),
        "reward": asdict(RewardConfig()),
        "oos_used_for_training": False,
        "checkpoint_rule": "episode-boundary checkpoint after every predeclared instance",
        "instance_selection": (
            "explicit predeclared training subset"
            if args.instance_ids is not None
            else "policy-seed shuffled training manifests"),
    }


def _checkpoint_path(out, completed_count):
    return out / "checkpoints" / f"episode_{completed_count:04d}.pt"


def _progress(out, run_config, completed_count, summaries, checkpoint, elapsed):
    return {
        "schema_version": 1,
        "completed": completed_count == run_config["episode_count"],
        "completed_episode_count": completed_count,
        "episode_count": run_config["episode_count"],
        "checkpoint": str(checkpoint.relative_to(out)),
        "checkpoint_sha256": _sha256(checkpoint),
        "elapsed_seconds": float(elapsed),
        "episodes": summaries,
    }


def _validate_resume(out, run_config, manifests):
    saved_config = _read_json(out / "training_configuration.json")
    if saved_config != run_config:
        raise ValueError("resume arguments do not exactly match training_configuration.json")
    progress = _read_json(out / "training_progress.json")
    completed = int(progress["completed_episode_count"])
    summaries = progress["episodes"]
    if not 0 <= completed <= len(manifests) or len(summaries) != completed:
        raise ValueError("training_progress.json has an invalid completed episode count")
    for index in range(completed):
        expected = manifests[index].stem
        if summaries[index].get("instance") != expected:
            raise ValueError("training progress does not match the frozen instance order")
        marker = out / "episodes" / expected / "COMPLETE.json"
        if not marker.exists() or _read_json(marker).get("completed") is not True:
            raise ValueError(f"completed episode marker is missing or invalid: {marker}")
    checkpoint = out / progress["checkpoint"]
    if not checkpoint.exists() or _sha256(checkpoint) != progress["checkpoint_sha256"]:
        raise ValueError("current resume checkpoint is missing or has the wrong SHA-256")
    return completed, summaries, checkpoint, float(progress.get("elapsed_seconds", 0.0))


def _quarantine_partial_episode(out, episode_out):
    if not episode_out.exists() or not any(episode_out.iterdir()):
        return
    quarantine = out / "interrupted_episodes"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = quarantine / f"{episode_out.name}_{stamp}"
    suffix = 1
    while destination.exists():
        destination = quarantine / f"{episode_out.name}_{stamp}_{suffix}"
        suffix += 1
    episode_out.rename(destination)
    print(f"[RESUME] quarantined incomplete episode output at {destination}", flush=True)


def _prune_old_checkpoints(out, current):
    for path in (out / "checkpoints").glob("episode_*.pt"):
        if path != current:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=("transformer", "mlp"), required=True)
    parser.add_argument("--component", choices=("M", "C", "I"), default="M",
                        help="Train a separate mutation, crossover, or initialization policy.")
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--evaluation-budget", type=int, required=True,
                        help="Predeclared real-objective evaluation budget per episode.")
    parser.add_argument(
        "--evaluation-mode", choices=("deterministic", "ccp100"),
        default="ccp100")
    parser.add_argument("--configuration", choices=tuple(SCENARIOS),
                        help="Restrict training to one fixed configuration/scene.")
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--gens", type=int, default=1_000)
    parser.add_argument("--max-instances", type=int,
                        help="Development-only cap; omit for the frozen formal training pass.")
    parser.add_argument(
        "--instance-ids", nargs="+",
        help=("Explicit predeclared ordered training subset. Mutually "
              "exclusive with --max-instances."))
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true",
                        help="Resume this exact run from its last committed episode checkpoint.")
    args = parser.parse_args()
    if args.evaluation_mode == "deterministic" and args.configuration is None:
        parser.error("deterministic stage requires one fixed --configuration")
    if not args.data.is_file():
        parser.error(f"network workbook does not exist: {args.data}")
    if not args.instances.is_dir():
        parser.error(f"instance directory does not exist: {args.instances}")
    if not args.normalization.is_file():
        parser.error(f"normalization file does not exist: {args.normalization}")
    if args.resume:
        if not args.out.exists() or not any(args.out.iterdir()):
            parser.error("--resume requires an existing non-empty output directory")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty unless --resume is supplied")
    try:
        manifests, records = _ordered_manifests(
            args.instances, args.policy_seed, args.max_instances,
            args.configuration, args.instance_ids)
    except ValueError as error:
        parser.error(str(error))
    config = (PPOConfig(rollout_steps=512, gae_horizon=256)
              if args.component == "I" else PPOConfig())
    normalizer = _normalizer(args.normalization)
    run_config = _run_config(args, config, manifests, records)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        try:
            completed, summaries, checkpoint, previous_elapsed = _validate_resume(
                args.out, run_config, manifests)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"cannot resume: {error}")
        trainer = PPOTrainer.load(
            checkpoint, PPOConfig, lambda *_: None,
            device=args.device, training=True)
        _prune_old_checkpoints(args.out, checkpoint)
        print(f"[RESUME] restored {completed}/{len(manifests)} episodes from {checkpoint}",
              flush=True)
    else:
        _atomic_json(args.out / "training_configuration.json", run_config)
        trainer = PPOTrainer(
            args.architecture, config, lambda *_: None,
            device=args.device, policy_seed=args.policy_seed, training=True,
            component=args.component,
            model_spec=_model_spec(args.component))
        completed, summaries, previous_elapsed = 0, [], 0.0
        checkpoint = _checkpoint_path(args.out, completed)
        trainer.save(checkpoint, metadata=run_config)
        _atomic_json(args.out / "training_progress.json", _progress(
            args.out, run_config, completed, summaries, checkpoint, previous_elapsed))

    if (args.out / "training_summary.json").exists():
        summary = _read_json(args.out / "training_summary.json")
        if summary.get("completed") is True and completed == len(manifests):
            print(f"[TRAIN] already complete: {completed}/{len(manifests)} episodes", flush=True)
            return
        parser.error("training_summary.json exists but is not a valid completed run")

    session_started = time.perf_counter()
    for index in range(completed, len(manifests)):
        manifest = manifests[index]
        seeds = episode_seeds(index, args.policy_seed)
        episode_out = args.out / "episodes" / manifest.stem
        if args.resume:
            _quarantine_partial_episode(args.out, episode_out)
        print(f"EPISODE {index + 1}/{len(manifests)} {manifest.stem}", flush=True)
        summary = run_episode(
            args.data, manifest, episode_out, trainer, normalizer,
            algorithm_seed=seeds["algorithm"], evaluation_budget=args.evaluation_budget,
            pop_size=args.pop, generations=args.gens, reward_config=RewardConfig(),
            evaluation_mode=args.evaluation_mode, component=args.component)
        summaries.append({"instance": manifest.stem, **summary})
        completed = index + 1
        previous_checkpoint = checkpoint
        checkpoint = _checkpoint_path(args.out, completed)
        trainer.save(checkpoint, metadata=run_config, overwrite=True)
        elapsed = previous_elapsed + time.perf_counter() - session_started
        _atomic_json(args.out / "training_progress.json", _progress(
            args.out, run_config, completed, summaries, checkpoint, elapsed))
        if previous_checkpoint != checkpoint and previous_checkpoint.exists():
            previous_checkpoint.unlink()
        per_episode = elapsed / completed
        eta_hours = per_episode * (len(manifests) - completed) / 3600.0
        print(f"[TRAIN] committed {completed}/{len(manifests)} "
              f"checkpoint={checkpoint.name} ETA={eta_hours:.2f}h", flush=True)

    final_checkpoint = args.out / "final_policy.pt"
    trainer.save(final_checkpoint, metadata=run_config, overwrite=True)
    elapsed = previous_elapsed + time.perf_counter() - session_started
    final_summary = {
        "completed": True, "component": args.component, "episodes": summaries,
        "checkpoint": str(final_checkpoint.relative_to(args.out)),
        "checkpoint_sha256": _sha256(final_checkpoint),
        "elapsed_seconds": elapsed,
        "total_actual_evaluations": sum(
            x["real_objective_evaluation_count"] for x in summaries),
    }
    _atomic_json(args.out / "training_summary.json", final_summary)
    print(f"[TRAIN] complete episodes={len(summaries)} checkpoint={final_checkpoint}",
          flush=True)


if __name__ == "__main__":
    main()
