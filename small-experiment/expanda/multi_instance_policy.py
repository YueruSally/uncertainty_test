"""Post-V1 location policies: the Rule mask is absolute for Rule and Learning."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import random

import joblib
import numpy as np

from multi_instance_features import feature_frame, feature_record


@dataclass
class Decision:
    chosen_index: int | None
    probabilities: list[float]
    scores: list[float | None]
    metadata: dict = field(default_factory=dict)


def baseline(candidates):
    values = [float(c["selection_probability"]) for c in candidates]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid original candidate probabilities")
    total = sum(values)
    if total <= 0:
        raise ValueError("zero candidate probability mass")
    return [x / total for x in values]


def rule_probabilities(candidates):
    original = baseline(candidates)
    eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
    mass = sum(original[i] for i in eligible)
    return ([original[i] / mass if i in eligible else 0.0
             for i in range(len(original))] if mass > 0 else [0.0] * len(original))


def _draw(probability, rng):
    return rng.choices(range(len(probability)), weights=probability, k=1)[0]


class RandomPolicy:
    name = "random-original"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, candidates, context=None):
        p = baseline(candidates)
        return Decision(_draw(p, self.rng), p, [None] * len(p),
                        {"no_eligible_candidate": False, "fallback": None})


class RulePolicy:
    name = "rule-masked-random-selection"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, candidates, context=None):
        p = rule_probabilities(candidates)
        if not any(p):
            return Decision(None, p, [None] * len(p),
                            {"no_eligible_candidate": True, "fallback": None})
        return Decision(_draw(p, self.rng), p, [None] * len(p),
                        {"no_eligible_candidate": False, "fallback": None})


class LearningPolicy:
    name = "rule-masked-model-reweighted-selection"

    def __init__(self, model_path, epsilon=.10, rng=None):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        self.epsilon = float(epsilon)
        self.rng = rng or random.Random()
        self.model_path = Path(model_path)
        self.artifact = joblib.load(self.model_path)
        if self.artifact.get("schema_version") != 3:
            raise ValueError("expected post-V1 model schema 3")
        if self.artifact.get("target") not in (
                "selection_survivor", "first_front_member", "parent_relation"):
            raise ValueError("unsupported model target")
        self.pipeline = self.artifact["pipeline"]

    def choose(self, candidates, context=None):
        p_rule = rule_probabilities(candidates)
        n = len(candidates)
        eligible = [i for i, p in enumerate(p_rule) if p > 0]
        if not eligible:
            return Decision(None, p_rule, [None] * n,
                            {"no_eligible_candidate": True, "fallback": None,
                             "epsilon": self.epsilon, "tv_distance": 0.0})
        if len(eligible) == 1:
            return Decision(eligible[0], p_rule, [None] * n,
                            {"no_eligible_candidate": False, "fallback": "single_candidate",
                             "epsilon": self.epsilon, "tv_distance": 0.0})
        import time
        start = time.perf_counter()
        frame = feature_frame([feature_record(candidates[i], context or {}) for i in eligible])
        proba = self.pipeline.predict_proba(frame)
        classes = list(self.pipeline.classes_)
        if self.artifact["target"] == "parent_relation":
            w = float(self.artifact["parent_relation_weight"])
            values = (proba[:, classes.index("dominates")]
                      + w * proba[:, classes.index("incomparable")])
        else:
            values = proba[:, classes.index(1)]
        scores = [None] * n
        for i, score in zip(eligible, values):
            scores[i] = float(score)
        weighted = [p_rule[i] * max(0.0, scores[i]) if i in eligible else 0.0
                    for i in range(n)]
        mass = sum(weighted)
        fallback = None
        if not math.isfinite(mass) or mass <= 0:
            model_p = p_rule
            fallback = "invalid_score_mass"
        else:
            model_p = [v / mass for v in weighted]
        p = [(1 - self.epsilon) * model_p[i] + self.epsilon * p_rule[i]
             for i in range(n)]
        p = [v / sum(p) for v in p]
        return Decision(_draw(p, self.rng), p, scores,
                        {"no_eligible_candidate": False, "fallback": fallback,
                         "epsilon": self.epsilon,
                         "tv_distance": .5 * sum(abs(a-b) for a, b in zip(p, p_rule)),
                         "different_argmax": int(np.argmax(p) != np.argmax(p_rule)),
                         "inference_seconds": time.perf_counter() - start})
