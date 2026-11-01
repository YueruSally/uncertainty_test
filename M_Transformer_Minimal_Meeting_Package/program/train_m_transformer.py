#!/usr/bin/env python3
"""Distil the zero-evaluation Balanced Replace teacher into the M Transformer."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random

import numpy as np


def _jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instance_key(value):
    """Use the same instance spelling as rollout directory names."""
    return str(value).removeprefix("train-").replace("-", "_")


def _teacher_target(row, feature_names):
    observation = row["observation"]
    features = np.asarray(observation["candidates"], dtype=float)
    eligible = np.asarray(observation["eligible"], dtype=bool)
    legal = np.flatnonzero(eligible)
    if len(legal) < 2:
        return None
    index = {name: feature_names.index(name) for name in (
        "batch_quantity", "replacement_delta_cost_per_teu",
        "replacement_delta_emission_per_teu",
        "replacement_delta_nominal_time_h",
        "replacement_max_lateness_h", "replacement_on_time_probability")}
    quantity = features[legal, index["batch_quantity"]]
    raw = np.column_stack((
        quantity * features[legal, index["replacement_delta_cost_per_teu"]],
        quantity * features[legal, index["replacement_delta_emission_per_teu"]],
        features[legal, index["replacement_delta_nominal_time_h"]],
        features[legal, index["replacement_max_lateness_h"]],
        -features[legal, index["replacement_on_time_probability"]],
    ))
    if not np.isfinite(raw).all():
        raise ValueError("non-finite Balanced teacher input")
    minima = raw.min(axis=0)
    spans = raw.max(axis=0) - minima
    spans[spans <= 1e-15] = 1.0
    scaled = (raw - minima) / spans
    scores = scaled[:, 0] + scaled[:, 1] + (
        scaled[:, 2] + scaled[:, 3] + scaled[:, 4]) / 3.0
    descriptors = row.get("candidates") or [{} for _ in observation["candidates"]]

    def tie_key(position):
        descriptor = descriptors[int(legal[position])]
        target = descriptor.get("target") or {}
        return (
            float(scores[position]),
            int(target.get("batch_id", 10**12)),
            str(descriptor.get("path_id")),
        )

    selected_position = min(range(len(legal)), key=tie_key)
    selected_index = int(legal[selected_position])
    return selected_index, scores.astype(float).tolist(), legal.astype(int).tolist()


def load_dataset(root):
    from transformer_ppo.observations import CANDIDATE_PATH_V4_FEATURES

    rows = []
    for path in sorted(Path(root).glob(
            "S0_*/seed_*/BOOTSTRAP_TRANSFORMER/mutation_policy_states.jsonl")):
        for state in _jsonl(path):
            if (state.get("decision_stage", "location") != "location"
                    or state.get("operator") != "replace"):
                continue
            target = _teacher_target(state, list(CANDIDATE_PATH_V4_FEATURES))
            if target is None:
                continue
            selected, scores, legal = target
            rows.append({
                "instance": _instance_key(state["instance_id"]),
                "generation": int(state["generation"]),
                "evaluation": int(state["evaluation"]),
                "observation": state["observation"],
                "teacher_action": selected,
                "teacher_scores": scores,
                "eligible_indices": legal,
                "source": str(path),
            })
    if not rows:
        raise ValueError(f"no usable M policy states under {root}")
    return rows


def _groups(rows, batch_size, rng):
    order = list(range(len(rows)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[index] for index in order[start:start + batch_size]]


def _state_key(row):
    return (
        f"{row['instance']}:g{row['generation']}:e{row['evaluation']}")


def _accuracy(torch, model, rows, device, batch_size):
    from transformer_ppo.observations import pad_observations

    model.eval()
    correct = 0
    losses = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            batch = pad_observations(
                [item["observation"] for item in group], device)
            logits, _ = model(batch)
            targets = torch.as_tensor(
                [item["teacher_action"] for item in group],
                dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            losses.append(float(loss.cpu()))
            correct += int((logits.argmax(dim=1) == targets).sum().cpu())
    return {
        "states": len(rows),
        "top_1_accuracy": correct / len(rows),
        "mean_cross_entropy": sum(losses) / max(1, len(losses)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-instances", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=62001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    if (args.epochs < 1 or args.batch_size < 1
            or args.early_stopping_patience < 1
            or args.learning_rate <= 0):
        parser.error(
            "epochs, batch size, patience and learning rate must be positive")

    import torch
    from transformer_ppo.config import PPOConfig
    from transformer_ppo.observations import pad_observations
    from transformer_ppo.ppo import PPOTrainer

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = load_dataset(args.data_root)
    validation_names = {
        _instance_key(value) for value in args.validation_instances}
    validation = [row for row in rows if row["instance"] in validation_names]
    training = [row for row in rows if row["instance"] not in validation_names]
    available_instances = sorted({row["instance"] for row in rows})
    print(
        f"[M-TEACHER-DATASET] states={len(rows)} "
        f"training={len(training)} validation={len(validation)} "
        f"available_instances={available_instances} "
        f"requested_validation={sorted(validation_names)}",
        flush=True)
    if not training or not validation:
        parser.error(
            "teacher dataset must contain both training and validation states; "
            f"available={available_instances}, "
            f"requested_validation={sorted(validation_names)}")

    trainer = PPOTrainer.load(
        args.checkpoint, PPOConfig, lambda *_: None,
        device=device, training=False)
    if (trainer.component != "M"
            or trainer.config.mutation_control != "replace_path"
            or trainer.config.candidate_representation != "path_v4"):
        parser.error(
            "checkpoint must be an M replace_path/path_v4 Transformer")
    trainer.config = replace(trainer.config, actor_policy="direct")
    trainer.model.actor_policy = "direct"
    model = trainer.model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    history = []
    best_validation = -1.0
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        numerical_fallback_groups = 0
        pending_groups = list(_groups(training, args.batch_size, rng))
        while pending_groups:
            group = pending_groups.pop(0)
            batch = pad_observations(
                [item["observation"] for item in group], device)
            targets = torch.as_tensor(
                [item["teacher_action"] for item in group],
                dtype=torch.long, device=device)
            logits, _ = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                keys = [_state_key(item) for item in group]
                if len(group) > 1:
                    pending_groups[0:0] = [[item] for item in group]
                    numerical_fallback_groups += 1
                    print(
                        "[M-TEACHER-TRAIN] split non-finite loss batch "
                        f"into {len(group)} microbatches: {keys}",
                        flush=True)
                    continue
                raise FloatingPointError(
                    "non-finite single-state Balanced-teacher imitation "
                    f"loss: {keys}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                keys = [_state_key(item) for item in group]
                optimizer.zero_grad(set_to_none=True)
                if len(group) > 1:
                    pending_groups[0:0] = [[item] for item in group]
                    numerical_fallback_groups += 1
                    print(
                        "[M-TEACHER-TRAIN] split non-finite gradient batch "
                        f"into {len(group)} microbatches: {keys}",
                        flush=True)
                    continue
                raise FloatingPointError(
                    "non-finite single-state Balanced-teacher imitation "
                    f"gradient: {keys}")
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
        train_metrics = _accuracy(
            torch, model, training, device, args.batch_size)
        validation_metrics = _accuracy(
            torch, model, validation, device, args.batch_size)
        record = {
            "epoch": epoch,
            "mean_training_loss": total_loss / max(1, steps),
            "numerical_fallback_groups": numerical_fallback_groups,
            "training": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"[M-TEACHER-TRAIN] epoch={epoch}/{args.epochs} "
            f"loss={record['mean_training_loss']:.6f} "
            f"train_top1={train_metrics['top_1_accuracy']:.4f} "
            f"valid_top1={validation_metrics['top_1_accuracy']:.4f} "
            f"fallback={numerical_fallback_groups}",
            flush=True)
        if validation_metrics["top_1_accuracy"] > best_validation + 1e-12:
            best_validation = validation_metrics["top_1_accuracy"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                "[M-TEACHER-TRAIN] early-stop "
                f"epoch={epoch} best_validation_top1={best_validation:.4f} "
                f"patience={args.early_stopping_patience}",
                flush=True)
            break
    model.load_state_dict(best_state)
    final_train = _accuracy(torch, model, training, device, args.batch_size)
    final_validation = _accuracy(
        torch, model, validation, device, args.batch_size)
    if not math.isfinite(final_validation["mean_cross_entropy"]):
        raise FloatingPointError("non-finite final validation metric")
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "final_policy.pt"
    trainer.save(checkpoint, metadata={
        "kind": "m_balanced_teacher_transformer",
        "teacher": "REPLACE_G10_JOINT_BALANCED",
        "training_state_count": len(training),
        "validation_state_count": len(validation),
        "validation_instances": sorted(validation_names),
        "seed": args.seed,
    })
    report = {
        "completed": True,
        "definition": (
            "path-v4 M Transformer behaviour-cloned from the deterministic "
            "Balanced joint batch/path selector"),
        "dataset_state_count": len(rows),
        "training_instances": sorted({row["instance"] for row in training}),
        "validation_instances": sorted({row["instance"] for row in validation}),
        "training": final_train,
        "validation": final_validation,
        "best_validation_top_1": best_validation,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "early_stopping_patience": args.early_stopping_patience,
        "stopped_early": stopped_early,
        "selected_epoch": max(
            history, key=lambda row: row["validation"]["top_1_accuracy"])[
                "epoch"],
        "checkpoint": str(checkpoint),
        "history": history,
    }
    (args.out / "TRAINING_COMPLETE.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
