"""Reward-only nondominated archive and exact normalized 3-D HV.

The archive is deliberately independent from NSGA-II environmental selection.
One policy action may produce more than one evaluated solution, so action
rewards are settled as an atomic batch against the archive state that existed
before any member of that batch was inserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


Point = Tuple[float, float, float]
ArchiveCandidate = Tuple[Sequence[float], bool, str]


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated(points: Iterable[Sequence[float]]) -> list[Point]:
    unique = sorted({tuple(float(x) for x in point) for point in points})
    return [p for p in unique if not any(_dominates(q, p) for q in unique if q != p)]


def _insert_nondominated(
        archive: Sequence[Point], additions: Iterable[Point]) -> list[Point]:
    """Incrementally insert points into an already nondominated archive."""
    updated = list(archive)
    for point in additions:
        if point in updated or any(_dominates(other, point) for other in updated):
            continue
        updated = [other for other in updated if not _dominates(point, other)]
        updated.append(point)
    return sorted(updated)


def _dominated_area_2d(points: Sequence[tuple[float, float]], ref_y: float,
                       ref_z: float) -> float:
    """Exact dominated area for minimization points in two dimensions."""
    clipped = sorted((y, z) for y, z in points if y < ref_y and z < ref_z)
    area = 0.0
    best_z = ref_z
    index = 0
    while index < len(clipped):
        y = clipped[index][0]
        while index < len(clipped) and clipped[index][0] == y:
            best_z = min(best_z, clipped[index][1])
            index += 1
        next_y = clipped[index][0] if index < len(clipped) else ref_y
        area += (next_y - y) * max(0.0, ref_z - best_z)
    return area


def _exact_hv_3d_nondominated(points: Sequence[Point], ref: Point) -> float:
    """Exact 3-D HV when ``points`` are already finite and nondominated."""
    clipped = sorted(
        point for point in points
        if all(point[index] < ref[index] for index in range(3))
    )
    if not clipped:
        return 0.0

    volume = 0.0
    active: list[tuple[float, float]] = []
    index = 0
    while index < len(clipped):
        left = clipped[index][0]
        while index < len(clipped) and clipped[index][0] == left:
            active.append((clipped[index][1], clipped[index][2]))
            index += 1
        right = clipped[index][0] if index < len(clipped) else ref[0]
        if right > left:
            volume += (right - left) * _dominated_area_2d(
                active, ref[1], ref[2])
    return float(volume)


def exact_hv_3d(points: Iterable[Sequence[float]], reference: Sequence[float]) -> float:
    """Exact dominated volume for minimization points in three dimensions.

    The sweep decomposes the x axis into slabs.  In each slab it computes the
    union area of origin-anchored rectangles in the y-z plane.
    """
    ref = tuple(float(x) for x in reference)
    if len(ref) != 3 or any(not np.isfinite(x) for x in ref):
        raise ValueError("finite three-dimensional reference required")
    clipped = nondominated(
        p for p in points
        if len(p) == 3 and all(np.isfinite(p)) and all(p[i] < ref[i] for i in range(3))
    )
    return _exact_hv_3d_nondominated(clipped, ref)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObjectiveNormalizer:
    minimum: Point
    maximum: Point
    reference: Point = (1.1, 1.1, 1.1)
    divisor: Point = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if any(not np.isfinite(x) for x in (
                *self.minimum, *self.maximum, *self.reference, *self.divisor)):
            raise ValueError("normalization values must be finite")
        if any(hi <= lo for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError("each normalization maximum must exceed its minimum")
        if any(value <= 0 for value in self.divisor):
            raise ValueError("objective divisors must be positive")

    def transform(self, objectives: Sequence[float]) -> Point:
        if len(objectives) != 3 or any(not np.isfinite(x) for x in objectives):
            raise ValueError("finite three-objective vector required")
        return tuple(
            (float(value) / divisor - lo) / (hi - lo)
            for value, divisor, lo, hi in zip(
                objectives, self.divisor, self.minimum, self.maximum)
        )  # type: ignore[return-value]

    def with_divisor(self, divisor: Sequence[float]):
        if len(divisor) != 3:
            raise ValueError("three objective divisors required")
        return ObjectiveNormalizer(
            self.minimum, self.maximum, self.reference,
            tuple(float(x) for x in divisor))  # type: ignore[arg-type]


@dataclass(frozen=True)
class RewardConfig:
    positive_hv_scale: float = 0.01
    floating_tolerance: float = 1e-12

    def validate(self) -> None:
        if not np.isfinite(self.positive_hv_scale) or self.positive_hv_scale <= 0:
            raise ValueError("positive_hv_scale must be finite and positive")
        if not np.isfinite(self.floating_tolerance) or self.floating_tolerance < 0:
            raise ValueError("floating_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class ArchiveUpdate:
    """Result of one atomic archive settlement."""

    delta_hv: float
    hv_before: float
    hv_after: float
    archive_size: int
    accepted_fingerprints: int


class RewardArchive:
    """Archive used only for reward; it never participates in NSGA-II selection."""

    def __init__(self, normalizer: ObjectiveNormalizer):
        self.normalizer = normalizer
        self._points: list[Point] = []
        self._fingerprints: set[str] = set()
        self._hypervolume = 0.0

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    @property
    def hypervolume(self) -> float:
        # The archive changes only through observe_many/reset.  Caching avoids
        # recomputing the same exact 3-D HV for reward-before, summaries, and
        # progress logs while preserving the exact value after every commit.
        return self._hypervolume

    def reset(self) -> None:
        """Start a new independent NSGA-II run."""
        self._points.clear()
        self._fingerprints.clear()
        self._hypervolume = 0.0

    def observe_many(self, candidates: Iterable[ArchiveCandidate]) -> ArchiveUpdate:
        """Atomically settle all feasible new solutions from one action.

        Each candidate is ``(objectives, feasible, fingerprint)``.  No staged
        candidate is visible to the archive until the joint HV increment has
        been calculated, which prevents child order from affecting a single
        action's total reward.
        """
        before = self.hypervolume
        staged_points: list[Point] = []
        staged_fingerprints: set[str] = set()
        for objectives, feasible, candidate_fingerprint in candidates:
            if (not feasible or candidate_fingerprint in self._fingerprints
                    or candidate_fingerprint in staged_fingerprints):
                continue
            try:
                point = self.normalizer.transform(objectives)
            except ValueError:
                continue
            staged_points.append(point)
            staged_fingerprints.add(candidate_fingerprint)

        updated_points = _insert_nondominated(self._points, staged_points)
        if updated_points == self._points:
            after = before
        else:
            after = _exact_hv_3d_nondominated(
                updated_points, self.normalizer.reference)
        raw_delta = after - before
        tolerance = max(1e-15, abs(before) * 1e-12)
        if raw_delta < -tolerance:
            raise RuntimeError("adding archive points reduced hypervolume")
        delta = 0.0 if raw_delta <= tolerance else float(raw_delta)

        # Commit only after the joint reward has been computed.
        self._points = updated_points
        self._fingerprints.update(staged_fingerprints)
        self._hypervolume = float(after)
        return ArchiveUpdate(
            delta_hv=delta,
            hv_before=float(before),
            hv_after=float(after),
            archive_size=len(self._points),
            accepted_fingerprints=len(staged_fingerprints),
        )

    def observe(self, objectives: Sequence[float], feasible: bool, fingerprint: str) -> float:
        """Compatibility wrapper for a one-solution non-policy update."""
        return self.observe_many([(objectives, feasible, fingerprint)]).delta_hv

    def summary(self) -> dict:
        if not self._points:
            return {"archive_size": 0, "archive_hv": 0.0,
                    "archive_min": [0.0] * 3, "archive_max": [0.0] * 3}
        values = np.asarray(self._points, dtype=float)
        return {"archive_size": len(self._points), "archive_hv": self.hypervolume,
                "archive_min": values.min(axis=0).tolist(),
                "archive_max": values.max(axis=0).tolist()}


def hv_reward(delta_hv: float, config: RewardConfig) -> float:
    """Pure positively scaled HV increment, without shaping or clipping."""
    config.validate()
    if not np.isfinite(delta_hv):
        raise ValueError("delta_hv must be finite")
    if delta_hv < -config.floating_tolerance:
        raise ValueError("delta_hv cannot be materially negative")
    return 0.0 if delta_hv <= config.floating_tolerance else float(
        delta_hv / config.positive_hv_scale)
