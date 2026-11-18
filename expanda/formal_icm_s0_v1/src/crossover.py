"""K8 complete crossover-plus-Base-mutation prescreening.

The deployable decision unit is a *pair* of complete offspring.  Candidate
zero reproduces Base's common-node cuts and all eight candidates receive the
same two Base mutation coin flips and the same per-child mutation RNG stream.
Only the selected pair is a logical B400 evaluation; Oracle-only evaluations
of the other seven pairs are reported separately.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import random
import time

import baseline_uncertainty as base
from multi_instance_logging import fingerprint

from transformer_ppo.archive import exact_hv_3d
from transformer_ppo.comparison_controllers import BaseBudgetController
from transformer_ppo.crossover import (
    apply_cut, finalize_children, prepare_common_node_crossover)
from transformer_ppo.crossover_rule import _candidate_proxy, _joint_single_override_plans
from initialization import (
    deterministic_proxy_rows, individual_batch_tokens)
from transformer_ppo.v2_controls import private_seed


METHODS = ("simple", "oracle", "collect_simple", "transformer")


def _pair_proxy_rows(candidate_pairs, batches, arcs):
    """Return deterministic pre-evaluation scores for complete child pairs."""
    flat = [child for pair in candidate_pairs for child in pair]
    child_rows = deterministic_proxy_rows(flat, batches, arcs)
    rows = []
    for index in range(len(candidate_pairs)):
        left, right = child_rows[2 * index:2 * index + 2]
        rows.append({
            "candidate_index": index,
            "score": float(left["score"] + right["score"]),
            "child_scores": [float(left["score"]), float(right["score"])],
            "cost": float(left["cost"] + right["cost"]),
            "emission": float(left["emission"] + right["emission"]),
            "time": float(max(left["time"], right["time"])),
            "capacity_excess": float(
                left["capacity_excess"] + right["capacity_excess"]),
            "lateness": float(left["lateness"] + right["lateness"]),
            "missing": int(left["missing"] + right["missing"]),
        })
    return rows


def _pair_batch_tokens(pair, batches):
    left = individual_batch_tokens(pair[0], batches)
    right = individual_batch_tokens(pair[1], batches)
    return [first + second for first, second in zip(left, right)]


def _pair_candidate_features(proxy_rows, roles):
    rows = []
    for index, (proxy, role) in enumerate(zip(proxy_rows, roles)):
        rows.append([
            float(proxy["score"]), float(proxy["child_scores"][0]),
            float(proxy["child_scores"][1]), float(proxy["cost"]),
            float(proxy["emission"]), float(proxy["time"]),
            float(proxy["capacity_excess"]), float(proxy["lateness"]),
            float(proxy["missing"]), index / 7.0,
            float(role == "base"), float(role == "directed_balanced"),
            float(role == "directed_safe"),
            float(role.startswith("private")),
            float(role == "base_duplicate"),
        ])
    return rows


def _single_override_actions(decisions):
    actions = []
    for cut_index, cut in enumerate(decisions):
        shadow = _candidate_proxy(cut.candidates[cut.shadow_base_index])
        for candidate_index, candidate in enumerate(cut.candidates):
            if candidate_index == cut.shadow_base_index:
                continue
            raw = _candidate_proxy(candidate)
            actions.append({
                "cut_index": cut_index,
                "candidate_index": candidate_index,
                "delta": tuple(value - reference
                               for value, reference in zip(raw, shadow)),
            })
    return actions


def _candidate_plans(decisions, run_id, event_index, pool_size=8):
    """Build a fixed K8 pool containing Base and directed alternatives."""
    base_plan = tuple(int(cut.shadow_base_index) for cut in decisions)
    plans = [base_plan]
    roles = ["base"]

    def add(plan, role):
        plan = tuple(map(int, plan))
        if plan not in plans:
            plans.append(plan); roles.append(str(role))

    directed, _ = _joint_single_override_plans(decisions, safe=False)
    add([row["candidate_index"] for row in directed], "directed_balanced")
    safe, _ = _joint_single_override_plans(decisions, safe=True)
    add([row["candidate_index"] for row in safe], "directed_safe")

    actions = _single_override_actions(decisions)
    dimension_names = ("cost", "emission", "time", "deadline", "uncertainty")
    for dimension, name in enumerate(dimension_names):
        if not actions:
            break
        action = min(actions, key=lambda row: (
            row["delta"][dimension], sum(row["delta"]),
            row["cut_index"], row["candidate_index"]))
        plan = list(base_plan)
        plan[action["cut_index"]] = action["candidate_index"]
        add(plan, name)

    generator = random.Random(private_seed(
        "C-K8-private", run_id, event_index))
    attempts = 0
    while len(plans) < pool_size and attempts < 256:
        attempts += 1
        plan = [generator.randrange(len(cut.candidates)) for cut in decisions]
        add(plan, f"private_{len(plans)}")

    # Rare crossover events have fewer than eight distinct legal full plans.
    # Keep K fixed without claiming artificial diversity; duplicate Base is a
    # conservative no-op and the unique count is logged explicitly.
    unique_count = len(plans)
    while len(plans) < pool_size:
        plans.append(base_plan); roles.append("base_duplicate")
    return plans[:pool_size], roles[:pool_size], unique_count


class CrossoverK8Controller(BaseBudgetController):
    """Choose one of eight complete crossover-plus-mutation child pairs."""

    controls_crossover = True
    controls_mutation = False

    def __init__(self, *args, selection_mode, reliable_options, batches, arcs,
                 pool_size=8, selection_policy=None,
                 defer_selected_pair_mutation=False, **kwargs):
        if selection_mode not in METHODS:
            raise ValueError(f"selection_mode must be one of {METHODS}")
        if int(pool_size) != 8:
            raise ValueError("the frozen C screen requires K=8")
        self.selection_mode = selection_mode
        if (selection_mode == "transformer") != (selection_policy is not None):
            raise ValueError(
                "selection_policy must be supplied only for transformer mode")
        self.selection_policy = selection_policy
        self.defer_selected_pair_mutation = bool(
            defer_selected_pair_mutation)
        if self.defer_selected_pair_mutation and selection_mode != "transformer":
            raise ValueError(
                "deferred C-K8 mutation requires transformer selection")
        self.reliable_options = dict(reliable_options)
        self.batches_for_proxy = tuple(batches)
        self.arcs_for_proxy = tuple(arcs)
        self.pool_size = int(pool_size)
        self.current_population = []
        self._active = None
        self.pool_events = 0
        self.eligible_pool_events = 0
        self.unique_candidate_total = 0
        self.candidate_objective_evaluations = 0
        self.hidden_candidate_objective_evaluations = 0
        self.candidate_construction_objective_evaluations = 0
        self.selection_counts = Counter()
        self.oracle_gain_total = 0.0
        self.simple_gain_total = 0.0
        self.feedback_rows = []
        super().__init__(*args, log_names=(
            "crossover_k8_candidates", "generation_summary"), **kwargs)

    def evaluate(self, function, ind, *args, **kwargs):
        before = self.evaluations
        result = super().evaluate(function, ind, *args, **kwargs)
        if self.generation < 0 and self.evaluations > before:
            self.current_population.append(ind)
        return result

    def flush_events(self, population):
        self.current_population = list(population)

    def crossover_common_node(
            self, parent1, parent2, batches, tt_dict, arc_lookup):
        if self._active is not None:
            raise RuntimeError("previous C-K8 pair was not settled")
        uncut1, uncut2, decisions = prepare_common_node_crossover(
            parent1, parent2, batches, tt_dict=tt_dict)
        self.total_attempts += 1
        self.pool_events += 1
        event_index = self.pool_events - 1
        plans, roles, unique_count = _candidate_plans(
            decisions, self.run_id, event_index, self.pool_size)
        candidate_pairs = []
        for plan in plans:
            child1, child2 = deepcopy(uncut1), deepcopy(uncut2)
            for cut, candidate_index in zip(decisions, plan):
                apply_cut(child1, child2, cut,
                          cut.candidates[candidate_index], tt_dict, arc_lookup)
            finalize_children(child1, child2)
            candidate_pairs.append((child1, child2))

        if unique_count < 2:
            self.no_candidate += 1
            return candidate_pairs[0]
        self.eligible_pool_events += 1
        self.unique_candidate_total += unique_count
        self._active = {
            "event_index": event_index,
            "generation": int(self.generation),
            "pairs": candidate_pairs,
            "roles": roles,
            "unique_count": unique_count,
            "fingerprints_before_mutation": [
                [fingerprint(child) for child in pair]
                for pair in candidate_pairs],
        }
        return candidate_pairs[0]

    @staticmethod
    def _mutate_without_evaluation(
            ind, batches, path_lib, tt_dict, arc_lookup, reliable_options):
        batch = random.choice(batches)
        operator = base.sample_operator()
        ok = base.apply_mutation_op(
            ind, operator, batch, path_lib, tt_dict, arc_lookup,
            reliable_options=reliable_options)
        if ok:
            base.repair_missing_allocations(ind, batches, path_lib)
        return operator, bool(ok)

    def _paired_base_mutation(self, candidate_pairs, position, batches,
                              path_lib, tt_dict, arc_lookup,
                              reliable_options, mutation_rate=None):
        # The outer coin and candidate-zero stream exactly reproduce Base.
        rate = base.MUTATION_RATE if mutation_rate is None else float(mutation_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("paired crossover mutation rate must be in [0, 1]")
        if random.random() >= rate:
            return {"applied": False, "operator": None, "success": False}
        state = random.getstate(); reference_post = None; reference = None
        for index, pair in enumerate(candidate_pairs):
            random.setstate(state)
            outcome = self._mutate_without_evaluation(
                pair[position], batches, path_lib, tt_dict, arc_lookup,
                reliable_options)
            if index == 0:
                reference = outcome; reference_post = random.getstate()
        random.setstate(reference_post)
        return {"applied": True, "operator": reference[0],
                "success": reference[1]}

    def _population_hv(self, extra_pair=None):
        individuals = list(self.current_population)
        if extra_pair is not None:
            individuals.extend(extra_pair)
        points = [self.normalizer.transform(ind.objectives)
                  for ind in individuals if ind.feasible]
        return exact_hv_3d(points, self.normalizer.reference)

    def _state_row(self, active, proxy_rows):
        return {
            "instance": self.run_id,
            "event_index": int(active["event_index"]),
            "slot_fraction": min(1.0, self.evaluations / max(1, self.budget)),
            "generation": int(active["generation"]),
            "population_points": [
                list(self.normalizer.transform(member.objectives))
                for member in self.current_population if member.feasible
            ],
            "candidate_features": _pair_candidate_features(
                active["proxy_rows"], active["roles"]),
            "candidate_batch_tokens": [
                _pair_batch_tokens(pair, self.batches_for_proxy)
                for pair in active["pairs"]
            ],
            "candidate_roles": list(active["roles"]),
            "candidate_fingerprints": [
                list(values) for values in
                active["fingerprints_after_mutation"]
            ],
        }

    def _commit_pre_evaluated(self, pair):
        if not self.has_budget(2):
            raise RuntimeError("logical objective evaluation budget exhausted")
        for ind in pair:
            self.evaluations += 1
            ind._learning_eval_key = (self.scenario_id, fingerprint(ind))
            ind._learning_eval_id = self.evaluations
            self._archive_pending.append(
                (ind.objectives, bool(ind.feasible), fingerprint(ind)))

    def process_crossover_pair(
            self, function, child1, child2, batches, path_lib, tt_dict,
            arc_lookup, arcs, waiting_cost, waiting_emission,
            reliable_options=None, paired_mutation_rate=None, **kwargs):
        active = self._active
        if active is None:
            return None
        if (child1 is not active["pairs"][0][0]
                or child2 is not active["pairs"][0][1]):
            # crossover_hybrid legitimately falls back to structural
            # crossover when the common-node result is empty.
            self._active = None
            return None
        pairs = active["pairs"]
        pre_mutation_pairs = (
            deepcopy(pairs) if self.defer_selected_pair_mutation else None)
        random_state = (
            random.getstate() if self.defer_selected_pair_mutation else None)
        mutations = [self._paired_base_mutation(
            pairs, position, batches, path_lib, tt_dict, arc_lookup,
            reliable_options, mutation_rate=paired_mutation_rate)
            for position in (0, 1)]
        if self.defer_selected_pair_mutation:
            # The frozen C model was trained on complete crossover followed by
            # paired Base mutation.  Construct that exact pre-evaluation view
            # for inference, then restore the stream and execute the selected
            # crossover pair through the separate M component exactly once.
            random.setstate(random_state)
        for pair in pairs:
            for child in pair:
                base.repair_missing_allocations(child, batches, path_lib)

        proxy_rows = _pair_proxy_rows(
            pairs, self.batches_for_proxy, self.arcs_for_proxy)
        gains = None
        if self.selection_mode in ("simple", "transformer"):
            active.update({
                "proxy_rows": proxy_rows,
                "fingerprints_after_mutation": [
                    [fingerprint(child) for child in pair] for pair in pairs],
            })
            if self.selection_mode == "simple":
                selected = min(range(len(pairs)), key=lambda index: (
                    proxy_rows[index]["score"], index != 1, index))
            else:
                started = time.perf_counter()
                selected = int(self.selection_policy.select(
                    self._state_row(active, proxy_rows)))
                self.inference_seconds += time.perf_counter() - started
                if not 0 <= selected < len(pairs):
                    raise ValueError("Transformer selected invalid C-K8 pair")
            if self.defer_selected_pair_mutation:
                chosen = pre_mutation_pairs[selected]
                for target, source in zip((child1, child2), chosen):
                    target.__dict__.clear()
                    target.__dict__.update(deepcopy(source.__dict__))
                self.total_effective += int(selected != 0)
                self.selection_counts[active["roles"][selected]] += 1
                active.update({
                    "selected_index": int(selected),
                    "proxy_rows": proxy_rows,
                    "gains": None,
                    "mutations": mutations,
                    "deferred_selected_pair_mutation": True,
                    "fingerprints_after_mutation": [
                        [fingerprint(child) for child in pair]
                        for pair in pairs],
                })
                # Returning None resumes the ordinary NSGA-II mutation stage,
                # where the combined controller invokes M on each child.
                return None
            chosen = pairs[selected]
            for child in chosen:
                self.evaluate(function, child, batches, arcs, tt_dict,
                              waiting_cost, waiting_emission, **kwargs)
            self.candidate_objective_evaluations += 2
            base_hv = self._population_hv()
            self.simple_gain_total += self._population_hv(chosen) - base_hv
        else:
            started = time.perf_counter()
            for pair in pairs:
                for child in pair:
                    function(child, batches, arcs, tt_dict,
                             waiting_cost, waiting_emission, **kwargs)
            self.objective_evaluation_seconds += time.perf_counter() - started
            self.candidate_objective_evaluations += 2 * len(pairs)
            self.hidden_candidate_objective_evaluations += 2 * (len(pairs) - 1)
            base_hv = self._population_hv()
            gains = [self._population_hv(pair) - base_hv for pair in pairs]
            if self.selection_mode == "oracle":
                selected = max(range(len(pairs)), key=lambda index: (
                    gains[index], sum(child.feasible for child in pairs[index]),
                    index == 0, -proxy_rows[index]["score"], -index))
            else:
                selected = min(range(len(pairs)), key=lambda index: (
                    proxy_rows[index]["score"], index != 1, index))
            chosen = pairs[selected]
            if self.selection_mode == "oracle":
                self.oracle_gain_total += gains[selected]
            else:
                self.simple_gain_total += gains[selected]
            self._commit_pre_evaluated(chosen)

        self.total_effective += int(selected != 0)
        self.selection_counts[active["roles"][selected]] += 1
        active.update({
            "selected_index": int(selected),
            "proxy_rows": proxy_rows,
            "gains": gains,
            "mutations": mutations,
            "fingerprints_after_mutation": [
                [fingerprint(child) for child in pair] for pair in pairs],
        })
        if self.selection_mode == "collect_simple":
            state = self._state_row(active, proxy_rows)
            state.update({
                "labels": [{
                    "index": index,
                    "objectives": [
                        [float(value) for value in child.objectives]
                        for child in pair
                    ],
                    "feasible": [bool(child.feasible) for child in pair],
                    "hv_contribution": float(gains[index]),
                    "fingerprints": list(
                        active["fingerprints_after_mutation"][index]),
                    "source": active["roles"][index],
                } for index, pair in enumerate(pairs)],
                "selected_index": int(selected),
                "simple_selected_index": int(selected),
                "additional_objective_evaluations": 14,
            })
            self.feedback_rows.append(state)
        return chosen

    def settle_crossover(self, child1, child2):
        active = self._active
        if active is None:
            return
        selected = active["selected_index"]
        if active.get("deferred_selected_pair_mutation"):
            active["pairs"][selected] = (child1, child2)
            active["fingerprints_after_mutation"][selected] = [
                fingerprint(child1), fingerprint(child2)]
        for index, (pair, role, proxy) in enumerate(zip(
                active["pairs"], active["roles"], active["proxy_rows"])):
            self.append("crossover_k8_candidates", {
                "event_index": active["event_index"],
                "generation": active["generation"],
                "candidate_index": index,
                "candidate_role": role,
                "unique_candidate_count": active["unique_count"],
                "fingerprints_before_mutation": (
                    active["fingerprints_before_mutation"][index]),
                "fingerprints_after_mutation": (
                    active["fingerprints_after_mutation"][index]),
                "proxy": proxy,
                "objectives": (
                    [[float(value) for value in child.objectives]
                     for child in pair]
                    if self.selection_mode in ("oracle", "collect_simple")
                    or index == selected
                    else None),
                "feasible": (
                    [bool(child.feasible) for child in pair]
                    if self.selection_mode in ("oracle", "collect_simple")
                    or index == selected
                    else None),
                "oracle_hv_gain": (
                    None if active["gains"] is None
                    else float(active["gains"][index])),
                "selected": index == selected,
                "selection_mode": self.selection_mode,
                "paired_base_mutations": active["mutations"],
            })
        self._active = None

    def finish_episode(self):
        if self._active is not None:
            raise RuntimeError("episode ended with an unsettled C-K8 pair")
        super().finish_episode()

    def summary(self):
        result = super().summary()
        result.update({
            "crossover_k8_selection_mode": self.selection_mode,
            "crossover_k8_pool_size": self.pool_size,
            "crossover_k8_pool_events": self.pool_events,
            "crossover_k8_eligible_pool_events": self.eligible_pool_events,
            "crossover_k8_unique_candidate_total": self.unique_candidate_total,
            "crossover_k8_selection_counts": dict(self.selection_counts),
            "crossover_k8_candidate_objective_evaluations": (
                self.candidate_objective_evaluations),
            "crossover_k8_hidden_candidate_objective_evaluations": (
                self.hidden_candidate_objective_evaluations),
            "crossover_k8_candidate_construction_objective_evaluations": (
                self.candidate_construction_objective_evaluations),
            "crossover_k8_oracle_gain_total": self.oracle_gain_total,
            "crossover_k8_simple_gain_total": self.simple_gain_total,
            "crossover_k8_physical_objective_evaluations": (
                self.evaluations + self.hidden_candidate_objective_evaluations),
        })
        return result

import copy
import numpy as np
import torch
from i_k8_surrogate import CandidateGainTransformer, pack_rows


def simple_index(row):
    """Reproduce C-K8 Simple using only pre-evaluation candidate features."""
    features = row.get("candidate_features", ())
    if len(features) != 8:
        raise ValueError("C-K8 residual requires eight candidate features")
    selected = min(range(8), key=lambda index: (
        float(features[index][0]), index != 1, index))
    recorded = row.get("simple_selected_index")
    if recorded is not None and int(recorded) != selected:
        raise ValueError("recorded C-K8 Simple decision is not reproducible")
    return selected


def model_order(row):
    selected = simple_index(row)
    return (selected, *(index for index in range(8) if index != selected))


def reorder_row(row):
    """Move this state's Simple choice to model index zero."""
    result = copy.deepcopy(row)
    order = model_order(row)
    for key in ("candidate_features", "candidate_batch_tokens", "labels",
                "candidate_fingerprints", "candidate_roles"):
        if key in result:
            values = result[key]
            if len(values) != 8:
                raise ValueError(f"C-K8 residual {key} must have length eight")
            result[key] = [values[index] for index in order]
    result["residual_original_order"] = list(order)
    result["simple_selected_index"] = 0
    return result


def original_index(row, residual_index):
    order = model_order(row)
    if not 0 <= int(residual_index) < len(order):
        raise ValueError("C-K8 residual index is invalid")
    return int(order[int(residual_index)])


def ensemble_predictions(models, rows, scaler, target_scale):
    ordered = [reorder_row(row) for row in rows]
    packed = pack_rows(ordered, scaler)
    values = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            values.append(model(*packed).numpy() * float(target_scale))
    predictions = np.asarray(values)
    if predictions.shape != (len(models), len(rows), 8):
        raise ValueError("C-K8 residual ensemble output shape changed")
    if not np.allclose(predictions[:, :, 0], 0.0, atol=1e-12):
        raise ValueError("Simple residual must be exactly zero")
    return predictions


def gate_decisions(predictions, uncertainty_weight, threshold):
    predictions = np.asarray(predictions, dtype=float)
    mean = predictions.mean(0)
    standard = predictions.std(0)
    conservative = mean - float(uncertainty_weight) * standard
    selected = []
    for values in conservative:
        best = int(np.argmax(values))
        selected.append(best if values[best] > float(threshold) else 0)
    return np.asarray(selected, dtype=int), mean, standard, conservative


class FrozenCResidualGate:
    """Frozen three-seed residual ensemble with C-K8 Simple fallback."""

    def __init__(self, bundle, digest_function):
        if bundle.get("protocol") != "c-k8-simple-residual-v1":
            raise ValueError("incorrect C-K8 residual bundle")
        self.scaler = bundle["scaler"]
        self.target_scale = float(bundle["target_scale"])
        self.uncertainty_weight = float(bundle["gate"]["uncertainty_weight"])
        self.threshold = float(bundle["gate"]["threshold"])
        self.models = []
        for filename in bundle["checkpoints"]:
            if digest_function(filename) != bundle["checkpoint_sha256"][filename]:
                raise ValueError(f"C residual checkpoint digest mismatch: {filename}")
            try:
                checkpoint = torch.load(
                    filename, map_location="cpu", weights_only=True)
            except TypeError as exc:
                if "weights_only" not in str(exc):
                    raise
                checkpoint = torch.load(filename, map_location="cpu")
            if (checkpoint.get("protocol") != "c-k8-simple-residual-model-v1"
                    or checkpoint.get("training_instances") != 200
                    or checkpoint.get("width") != 16):
                raise ValueError("C residual checkpoint identity changed")
            model = CandidateGainTransformer(
                checkpoint["batch_feature_count"],
                checkpoint["candidate_feature_count"],
                checkpoint["context_feature_count"], width=16)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("C residual gate requires three models")

    def select(self, row):
        predictions = ensemble_predictions(
            self.models, [row], self.scaler, self.target_scale)
        selected, _, _, _ = gate_decisions(
            predictions, self.uncertainty_weight, self.threshold)
        return original_index(row, int(selected[0]))
