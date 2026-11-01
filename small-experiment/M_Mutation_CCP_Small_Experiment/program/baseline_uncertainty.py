#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import pathlib
import time
import argparse
import os
from statistics import NormalDist
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path as FSPath
import json as _json

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(FSPath(".cache/matplotlib").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ════════════════════════════════════════════════════════
# Global settings
# ════════════════════════════════════════════════════════

TIME_BUCKET_H = 24.0

CHINA_REGIONS   = {"CN"}
EUROPE_REGIONS  = {"EE", "WE"}
TRANSIT_REGIONS = {"KZ", "KG", "UZ", "RU", "BY"}

CORRIDOR_ORDER: Dict[str, int] = {"CN": 0, "CA": 1, "RU": 2, "EE": 3, "WE": 4}

CHINA_BORDER_NODES: set = {"Erenhot", "Manzhouli", "Khorgos", "Lianyungang",
                            "Chongqing", "Yiwu"}
NODE_GROUP: Dict[str, str] = {}

# ════════════════════════════════════════════════════════
# [V1 WIRING] Congestion-node classification
# ────────────────────────────────────────────────────────
# NOTE: CHINA_BORDER_NODES above is used ONLY for routing
# monotonicity (china_border_monotone_ok) — do NOT repurpose it.
# The sets below control where border delay & capacity apply.
#
# Break-of-gauge / transit border nodes = the true 1435↔1520mm
# change points + shared chokepoint. ONLY these get fixed border
# delay AND a node capacity soft-constraint. Ordinary EU
# destinations (Berlin, Hamburg, Duisburg, Rotterdam, Prague…)
# are deliberately EXCLUDED even though the data generator wrongly
# tagged them IsBorderNode=1 with 40.6h delay + capacity 500.
BREAK_OF_GAUGE_NODES: set = {
    "Khorgos", "Alashankou", "Dostyk", "Altynkol",
    "Manzhouli", "Zabaykalsk", "Erenhot",
    "Brest", "Malaszewicze",
}
# Sea ports — candidate congestion nodes. Wired but only active
# when a positive BorderCapacity_TEUday is supplied for them
# (current data has none, so they stay inactive in v1).
PORT_CONGESTION_NODES: set = {
    "Shanghai", "Ningbo", "Tianjin", "Qingdao", "Dalian",
    "Rotterdam", "Hamburg", "Antwerp",
}
BOTTLENECK_NODES: set = set(BREAK_OF_GAUGE_NODES) | set(PORT_CONGESTION_NODES)

# Filled at load time from Node_Border (BorderCapacity / BackgroundFlow)
BORDER_CAPACITY: Dict[str, float] = {}
BACKGROUND_FLOW: Dict[str, float] = {}

# Soft-penalty weight for exceeding node/border daily capacity
PEN_BORDER_CAP_EXCESS_PER_TEU = 5e7

HARD_TIME_WINDOW = False

PEN_MISS_TT            = 5e7
PEN_MISS_ALLOC         = 1e9
PEN_CAP_EXCESS_PER_TEU = 5e7

WAITING_COST_PER_TEU_HOUR_DEFAULT    = 0.8
WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT = 320.50

WAITING_EMISSION_SCOPE = "timetable-induced schedule waiting"
WAITING_EMISSION_PARAMETERISATION = "common across road/rail/water"
WAITING_EMISSION_SOURCE_NOTE = (
    "literature-informed common modelling assumption based on "
    "Lomotko et al. (2023)")
WAITING_CARBON_REGION_BASIS = "waiting-node region"
WAITING_CARBON_MODE_BASIS = "outgoing scheduled mode"


def waiting_emission_configuration(
    wait_emission_g_per_teu_h: float =
        WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT,
) -> Dict[str, Any]:
    """Shared formal Work-2 waiting-emission configuration."""
    return {
        "waiting_emission_gCO2_per_TEU_h":
            float(wait_emission_g_per_teu_h),
        "waiting_emission_scope": WAITING_EMISSION_SCOPE,
        "waiting_emission_parameterisation":
            WAITING_EMISSION_PARAMETERISATION,
        "waiting_emission_source_note": WAITING_EMISSION_SOURCE_NOTE,
        "carbon_cost_waiting_emission": True,
        "waiting_carbon_region_basis": WAITING_CARBON_REGION_BASIS,
        "waiting_carbon_mode_basis": WAITING_CARBON_MODE_BASIS,
    }


def print_waiting_emission_configuration(
    wait_emission_g_per_teu_h: float =
        WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT,
) -> None:
    print("[CONFIG] Waiting emission = "
          f"{wait_emission_g_per_teu_h:.2f} gCO2/TEU/hour")
    print("[CONFIG] Waiting emissions included in regional carbon cost = True")

# ── GA hyper-parameters  [TUNED Stage 2 → G7] ────────────
CROSSOVER_RATE  = 0.90       # ← Stage 2 best (G7: pc=0.90, pm=0.15)
MUTATION_RATE   = 0.15       # ← Stage 2 best

W_ADD     = 0.20
W_DEL     = 0.20
W_MOD     = 0.20
W_MODE    = 0.20
W_REPLACE = 0.20
OPS       = ["add", "del", "mod", "mode", "replace"]
ACTIVE_MUTATION_LOGGER = None  # Enabled only by run_learning_ccp100.py.

# ── Path library  [TUNED Stage 3] ────────────────────────
PATHS_TOPK_PER_CRITERION = 15        # ← Stage 3 best
PATH_LIB_CAP_TOTAL       = 45        # ← TopK × 3
DFS_MAX_PATHS_PER_OD     = 200       # ← Stage 3 best

CROSSOVER_SEGMENT_PROB = 0.50

# ── Feasibility boost ────────────────────────────────────
MIN_FEASIBLE_SOLUTIONS       = 10
FEASIBLE_BOOST_ROUNDS        = 20
FEASIBLE_BOOST_MUTATION_RATE = 0.60
FEASIBLE_BOOST_TOPK_PARENTS  = 10

# Feasibility-first search. A fixed fraction of the initial population is
# built from structurally valid paths; on-time probability and maximum
# lateness are ordering diagnostics only. A min-conflicts search then
# coordinates arc and border capacities. This changes only search strategy.
FEASIBILITY_SEED_FRACTION = 0.30
FEASIBILITY_SEARCH_RESTARTS = 40
FEASIBILITY_SEARCH_ITERATIONS = 250

# ── Metrics ──────────────────────────────────────────────
HV_EVERY     = 5
HV_SAMPLES   = 50000
METRIC_EVERY = 5

PSTAR_TAIL_GENS   = 30
PSTAR_CAP_PER_GEN = 40
PSTAR_MAX_TOTAL   = 50000

HV_REF_NORM = (1.2, 1.2, 1.2)
HV_MC_SEED  = 12345

DEFAULT_LATE_PENALTY_USD_PER_TEU_H = 6.25
DEFAULT_LATE_PENALTY_USD_PER_TEU_DAY = (
    DEFAULT_LATE_PENALTY_USD_PER_TEU_H * 24.0
)
DEFAULT_PENALTY_PER_TEU_H = DEFAULT_LATE_PENALTY_USD_PER_TEU_H
PAYLOAD_TONNES_PER_TEU = 10.0

# Hardcoded operating speeds. These are a LAST-RESORT FALLBACK only, used
# solely when a mode is missing a positive value from both an explicit CLI
# override and the workbook's Mode_Speeds sheet. See
# resolve_effective_speed_map() and load_network_from_extended() for the
# full CLI-override > workbook > fallback precedence. Do not read this dict
# directly anywhere else -- use the returned/EFFECTIVE_MODE_SPEED_KMH map.
DEFAULT_MODE_SPEED_KMH = {
    "road": 40.0,
    "rail": 50.0,
    "water": 28.0,
}
# Populated from --road-speed-kmh/--rail-speed-kmh/--water-speed-kmh; a
# value stays None unless the user explicitly passed that flag, so the
# precedence resolver can tell "explicitly requested" apart from "unset".
CLI_MODE_SPEED_OVERRIDE_KMH: Dict[str, Optional[float]] = {
    "road": None, "rail": None, "water": None,
}
# Set once by load_network_from_extended(); the single source of truth for
# the nominal speed actually applied to every Arc this run. Read this (or
# the function's return value), never DEFAULT_MODE_SPEED_KMH, downstream.
EFFECTIVE_MODE_SPEED_KMH: Dict[str, float] = dict(DEFAULT_MODE_SPEED_KMH)
EFFECTIVE_MODE_SPEED_SOURCE: Dict[str, str] = {
    m: "hardcoded_fallback" for m in DEFAULT_MODE_SPEED_KMH
}

DEFAULT_BORDER_EVENT_DATA_FILE = str(
    FSPath(__file__).resolve().parent / "data" / "border_crossing_events.csv"
)
# Filled from --border-event-data before scenarios are generated.
BORDER_EVENT_DEFINITIONS: Dict[Tuple[str, str, str], "BorderEventDefinition"] = {}

NUM_OBJ = 3  # Cost, Emission, Time

# ── Stochastic extension: frozen scenarios + empirical CCP quantiles ────
STOCHASTIC_EVAL = True
MC_SCENARIOS = 200
MC_BASE_SEED = 1000003
CONFIDENCE_COST = 0.90
CONFIDENCE_EMISSION = 0.90
CONFIDENCE_TIME = 0.90

# ── Evaluation mode: which risk metric reduces the per-scenario objective
# arrays to the three fitness values NSGA-II optimises. One of
# "deterministic" / "ev" / "ccp". See aggregate_scenario_objectives().
RISK_METRIC = "ccp"

# Human-readable, mode-specific description for scenario_manifest.json's
# "fitness_evaluation" field -- must never describe EV or deterministic
# runs as CCP/empirical-order-statistics evaluation.
FITNESS_EVALUATION_DESCRIPTION = {
    "deterministic": (
        "single_nominal_scenario (S=1, all stochastic multipliers=1.0; "
        "no Monte Carlo sampling)"),
    "ev": (
        "direct_expected_inputs (uncertain travel multipliers fixed at "
        "their model expectation 1.0 and border disturbances fixed at "
        "their model mean; S=1, no Monte Carlo sample mean)"),
    "ccp": (
        "frozen_scenarios_empirical_order_statistics (separate quantile "
        "objectives for per-scenario cost/emission/time arrays; no "
        "punctuality chance constraint in the current implementation)"),
}

MODE_TIME_CV = {
    "road": 0.15,
    "rail": 0.10,
    "water": 0.20,
}

MODE_TIME_MAX_FACTOR = {
    "road": 2.0,
    "rail": 1.8,
    "water": 2.5,
}

BORDER_DELAY_CV = {
    "road": 0.45,
    "rail": 0.60,
    # Maritime/port border-delay CV. No empirical maritime border-delay
    # MEAN exists anywhere in the repository (Node_Border.BorderDelay_water_h
    # is 0 in every row of every workbook; border_crossing_events.csv has no
    # water row) -- this CV is a prototype assumption with nothing to
    # multiply against yet. build_scenario_set() already produces zero
    # variance for any zero-mean border event, so stochastic maritime
    # border delay stays at zero until a sourced mean is added; see the
    # [WARN] in build_scenario_set() that fires if that ever changes.
    "water": 0.40,
}
BORDER_DELAY_MAX_FACTOR = 3.5

# Generated once after the input data are loaded.  Fitness evaluation only
# reads this object, so Monte Carlo draws never occur inside NSGA-II.
ACTIVE_SCENARIO_SET: Optional["ScenarioSet"] = None
_PATH_SCENARIO_CACHE: Dict[Any, "PathScenarioResult"] = {}


# ════════════════════════════════════════════════════════
# Corridor constraint helpers
# ════════════════════════════════════════════════════════

def china_border_monotone_ok(nodes: List[str], node_region: Dict[str, str]) -> bool:
    passed_border = False
    left_china    = False
    start_is_border = (len(nodes) > 0 and nodes[0] in CHINA_BORDER_NODES)
    for i, n in enumerate(nodes):
        r        = str(node_region.get(n, "")).strip()
        in_china  = (r in CHINA_REGIONS)
        is_border = (n in CHINA_BORDER_NODES)
        if left_china and in_china:
            return False
        if in_china:
            if passed_border:
                return False
            if is_border and not (i == 0 and start_is_border):
                passed_border = True
        else:
            left_china = True
    return True


def region_monotone_ok(nodes: List[str], node_region: Dict[str, str]) -> bool:
    max_level = -1
    for n in nodes:
        grp   = NODE_GROUP.get(n, "")
        level = CORRIDOR_ORDER.get(grp, -1)
        if level < 0:
            continue
        if level < max_level:
            return False
        max_level = level
    return True


# ════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════

def normalize_mode(mode_raw: str) -> str:
    m = str(mode_raw).strip().lower()
    if m in {"railway", "rail"}:      return "rail"
    if m in {"road", "truck"}:        return "road"
    if m in {"water", "ship", "sea"}: return "water"
    return m


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x): return default
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return default


def capped_lognormal_mean(mu: float, sigma: float, cap_factor: float) -> float:
    """Return E[min(exp(N(mu, sigma^2)), cap_factor)]."""
    if cap_factor <= 0.0:
        raise ValueError("cap_factor must be positive")
    if sigma <= 1e-12:
        return min(math.exp(mu), cap_factor)
    log_cap = math.log(cap_factor)
    normal = NormalDist()
    below_first_moment = (
        math.exp(mu + 0.5 * sigma * sigma)
        * normal.cdf((log_cap - mu - sigma * sigma) / sigma)
    )
    tail_probability = 1.0 - normal.cdf((log_cap - mu) / sigma)
    return below_first_moment + cap_factor * tail_probability


def calibrated_capped_lognormal_mu(cv: float, cap_factor: float) -> float:
    """Choose mu so the capped multiplier has mean one and cap unchanged."""
    if cap_factor < 1.0:
        raise ValueError("cap_factor must be at least 1 for a mean-one multiplier")
    sigma = math.sqrt(math.log1p(max(0.0, cv) ** 2))
    if sigma <= 1e-12 or cap_factor <= 1.0 + 1e-12:
        return 0.0

    lower = -50.0
    upper = math.log(cap_factor) + 50.0
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        if capped_lognormal_mean(midpoint, sigma, cap_factor) < 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def sample_capped_mean_one_lognormal(
    size: int,
    cv: float,
    cap_factor: Optional[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw mean-one lognormal multipliers, optionally with a hard cap.

    ``cap_factor=None`` implements the standard untruncated parameterisation
    used by the transport-time uncertainty specification:
    ``sigma=sqrt(log(1+CV^2))`` and ``mu=-sigma^2/2``.  A finite cap retains
    the historical mean-calibrated truncated distribution used for border
    delays and legacy experiments.
    """
    if size <= 0:
        return np.empty(0, dtype=float)
    if cv <= 1e-12:
        return np.ones(size, dtype=float)
    sigma = math.sqrt(math.log1p(cv * cv))
    if cap_factor is None or not math.isfinite(float(cap_factor)):
        return rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=size)
    if cap_factor <= 1.0 + 1e-12:
        return np.ones(size, dtype=float)
    mu = calibrated_capped_lognormal_mu(cv, cap_factor)
    raw = rng.lognormal(mean=mu, sigma=sigma, size=size)
    return np.minimum(raw, cap_factor)


def empirical_ccp_quantile(values: np.ndarray, confidence: float) -> float:
    """Order-statistic CCP value: sorted[ceil(confidence*S)-1]."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("inf")
    confidence = min(1.0, max(0.0, float(confidence)))
    ordered = np.sort(arr)
    index = max(0, min(arr.size - 1, math.ceil(confidence * arr.size) - 1))
    return float(ordered[index])


def aggregate_scenario_objectives(
    cost_s: np.ndarray,
    emission_s: np.ndarray,
    makespan_s: np.ndarray,
    scenario_cost_mean: float,
    scenario_emission_mean: float,
    scenario_time_mean: float,
    mode: str,
    confidence_cost: float,
    confidence_emission: float,
    confidence_time: float,
) -> Tuple[float, float, float]:
    """Reduce the shared per-scenario objective arrays to fitness values.

    This is the ONLY point where deterministic / ev / ccp modes diverge.
    Scenario generation (build_scenario_set) and scenario-specific
    simulation (simulate_path_over_scenarios) are identical and shared
    across all three modes -- nothing about them is duplicated here.

    - "deterministic": scenario_set is expected to have been forced to a
      single nominal scenario (stochastic=False, size=1) upstream, so the
      arrays hold exactly one element each; that element is returned
      directly rather than relying on it being numerically equal to a
      mean-of-one (explicit, not an implicit coincidence).
    - "ev": returns the already-computed scenario_cost_mean /
      scenario_emission_mean / scenario_time_mean diagnostics unchanged --
      np.mean(cost_s) etc. are NOT recomputed here. Because the economic
      lateness penalty is already added into cost_s per scenario (see
      evaluate_individual), the EV cost mean automatically includes
      expected lateness cost; no separate lateness-risk term is added.
    - "ccp": calls the existing, unmodified empirical_ccp_quantile per
      array -- current behaviour, byte-for-byte unchanged.
    """
    if mode == "deterministic":
        return float(cost_s[0]), float(emission_s[0]), float(makespan_s[0])
    if mode == "ev":
        return (float(scenario_cost_mean), float(scenario_emission_mean),
                float(scenario_time_mean))
    if mode == "ccp":
        return (
            empirical_ccp_quantile(cost_s, confidence_cost),
            empirical_ccp_quantile(emission_s, confidence_emission),
            empirical_ccp_quantile(makespan_s, confidence_time),
        )
    raise ValueError(
        f"Unknown risk_metric mode {mode!r}; expected one of "
        "'deterministic', 'ev', 'ccp'.")


def parse_distance_km(x) -> float:
    s       = str(x)
    cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    return float(cleaned) if cleaned else 0.0


def norm_region(x: str) -> str:
    s  = str(x).strip()
    if not s or s.lower() in {"nan", "none", ""}: return ""
    sl = s.lower()
    if sl in {"china", "prc", "chn"}:                return "CN"
    if sl in {"we", "west europe", "western europe"}: return "WE"
    if sl in {"ee", "east europe", "eastern europe"}: return "EE"
    return s.upper()


def unique_objective_tuples(objs, tol=1e-9):
    out = []
    for o in objs:
        dup = any(all(abs(o[i] - p[i]) <= tol for i in range(NUM_OBJ)) for p in out)
        if not dup:
            out.append(o)
    return out


def _is_bad_text_token(s: str) -> bool:
    if s is None: return True
    t = str(s).strip()
    return t == "" or t.startswith("...")


def _ffill_nan(arr: np.ndarray) -> np.ndarray:
    x = np.array(arr, dtype=float).copy()
    if x.size == 0: return x
    finite_idx = np.where(np.isfinite(x))[0]
    if finite_idx.size == 0: return x
    x[~np.isfinite(x)] = np.nan
    first = finite_idx[0]
    if first > 0:
        x[:first] = x[first]
    for i in range(1, len(x)):
        if np.isnan(x[i]) and np.isfinite(x[i - 1]):
            x[i] = x[i - 1]
    return x


def _finite_points_array(pts):
    if not pts:
        return np.empty((0, NUM_OBJ), dtype=float)
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != NUM_OBJ:
        return np.empty((0, NUM_OBJ), dtype=float)
    return arr[np.all(np.isfinite(arr), axis=1)]


# ════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════

@dataclass
class Arc:
    from_node: str
    to_node: str
    mode: str
    distance: float
    capacity: float
    cost_per_teu_km: float
    emission_per_teu_km: float
    speed_kmh: float
    from_region: str = ""
    to_region:   str = ""
    is_border_arc: bool = False


@dataclass
class TimetableEntry:
    from_node: str
    to_node:   str
    mode:      str
    frequency_per_week: float
    first_departure_hour: float
    headway_hours: float
    travel_time_h: Optional[float] = None
    legacy_time_value: Optional[float] = None


@dataclass(frozen=True)
class BorderEventDefinition:
    """One directional border crossing event b=(exit, entry, mode)."""
    exit_node: str
    entry_node: str
    mode: str
    mean_delay_h: float
    exit_mean_h: Optional[float] = None
    entry_mean_h: Optional[float] = None
    source: str = ""
    source_year: Optional[int] = None


@dataclass
class Batch:
    batch_id:  int
    origin:    str
    destination: str
    quantity:  float
    ET:        float
    LT:        float
    penalty_per_teu_h: float = DEFAULT_PENALTY_PER_TEU_H
    max_late_h: Optional[float] = None


@dataclass
class Path:
    path_id:   int
    origin:    str
    destination: str
    nodes:  List[str]
    modes:  List[str]
    arcs:   List[Arc]
    base_cost_per_teu:     float
    base_emission_per_teu: float
    base_travel_time_h:    float

    def __eq__(self, other):
        if not isinstance(other, Path): return NotImplemented
        return self.nodes == other.nodes and self.modes == other.modes

    def __hash__(self):
        return hash((tuple(self.nodes), tuple(self.modes)))


@dataclass
class PathAllocation:
    path:  Path
    share: float

    def __repr__(self):
        chain = ""
        for i, node in enumerate(self.path.nodes[:-1]):
            chain += f"{node}--({self.path.modes[i]})-->"
        chain += self.path.nodes[-1]
        return f"\n    {{ Structure: [{chain}], Share: {self.share:.2%} }}"


# [NSGA-II] Individual carries rank + crowding_distance
@dataclass(eq=False)
class Individual:
    od_allocations: Dict[Tuple[str, str, int], List[PathAllocation]] = field(default_factory=dict)
    objectives:     Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    penalty:        float = 0.0
    feasible:       bool  = False
    feasible_hard:  bool  = False
    normalized_violation: float = float("inf")
    vio_breakdown:  Dict[str, float] = field(default_factory=dict)
    batch_on_time_prob: Dict[int, float] = field(default_factory=dict)
    batch_max_lateness_h: Dict[int, float] = field(default_factory=dict)
    cost_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    emission_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    makespan_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    linehaul_carbon_cost_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float))
    waiting_carbon_cost_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float))
    carbon_cost_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float))
    ev_objectives: Tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
    rank:               int   = 0           # [NSGA-II] non-dominated front index (0 = best)
    crowding_distance:  float = 0.0         # [NSGA-II] crowding distance


@dataclass(frozen=True)
class ScenarioSet:
    """A frozen finite-support approximation of the random environment."""
    size: int
    seed: int
    travel_multiplier: Dict[Tuple[str, str, str], np.ndarray]
    border_delay_h: Dict[Tuple[str, str, str], np.ndarray]
    arc_border_event: Dict[Tuple[str, str, str], Tuple[str, str, str]]
    border_event_mean_h: Dict[Tuple[str, str, str], float]
    stochastic: bool = True

    def travel(self, arc: Arc, scenario: int) -> float:
        key = (arc.from_node, arc.to_node, arc.mode)
        values = self.travel_multiplier.get(key)
        return float(values[scenario]) if values is not None else 1.0

    def border_event_for_arc(
        self, arc: Arc
    ) -> Optional[Tuple[str, str, str]]:
        return self.arc_border_event.get(
            (arc.from_node, arc.to_node, arc.mode))

    def border_for_arc(
        self, arc: Arc, scenario: int
    ) -> Tuple[Optional[Tuple[str, str, str]], float]:
        event_key = self.border_event_for_arc(arc)
        if event_key is None:
            return None, 0.0
        values = self.border_delay_h.get(event_key)
        if values is not None:
            return event_key, float(values[scenario])
        return event_key, float(self.border_event_mean_h.get(event_key, 0.0))


@dataclass
class PathScenarioResult:
    arrival_h: np.ndarray
    schedule_wait_h: Dict[str, np.ndarray]
    border_delay_h: Dict[str, np.ndarray]
    missing_timetable: bool = False


@dataclass
class ReliablePathOption:
    """One structurally feasible single-path batch assignment.

    The historical name is retained for API compatibility; reliability KPIs
    are diagnostic only and never screen this option.
    """
    path: Path
    on_time_probability: float
    max_lateness_h: float
    resources: Dict[Any, float]


# ════════════════════════════════════════════════════════
# Merge & normalise shares
# ════════════════════════════════════════════════════════

def merge_and_normalize(allocs: List[PathAllocation]) -> List[PathAllocation]:
    if not allocs: return []
    merged: Dict[Path, float] = {}
    for a in allocs:
        merged[a.path] = merged.get(a.path, 0.0) + float(a.share)
    unique_allocs = [PathAllocation(path=p, share=s) for p, s in merged.items()]
    total = sum(a.share for a in unique_allocs)
    if total <= 1e-12:
        avg = 1.0 / max(1, len(unique_allocs))
        for a in unique_allocs: a.share = avg
    else:
        for a in unique_allocs: a.share /= total
    filtered = [a for a in unique_allocs if a.share > 0.05]
    if not filtered:
        best = max(unique_allocs, key=lambda a: a.share)
        best.share = 1.0
        return [best]
    total2 = sum(a.share for a in filtered)
    if abs(total2 - 1.0) > 1e-9:
        for a in filtered: a.share /= total2
    return filtered


# The two extra pairs are retained only as workbook-derived fallbacks because
# they occur in the network but are not listed in the paper's Table II.
WORKBOOK_BORDER_PAIR_FALLBACKS = (
    ("Manzhouli", "Zabaykalsk", "rail"),
    ("Brest", "Malaszewicze", "rail"),
)


def load_border_event_definitions(
    filename: str,
) -> Dict[Tuple[str, str, str], BorderEventDefinition]:
    """Load auditable directional border-event means from a CSV file."""
    path = FSPath(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Border-event data file not found: {path}")
    df = pd.read_csv(path)
    required = {"ExitNodeEN", "EntryNodeEN", "Mode"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Border-event data missing required columns: {sorted(missing)}")

    definitions: Dict[Tuple[str, str, str], BorderEventDefinition] = {}
    for _, row in df.iterrows():
        exit_node = str(row.get("ExitNodeEN", "")).strip()
        entry_node = str(row.get("EntryNodeEN", "")).strip()
        mode = normalize_mode(row.get("Mode", ""))
        if not exit_node or not entry_node or not mode:
            continue
        exit_mean = safe_float(row.get("ExitMean_h"), default=None)
        entry_mean = safe_float(row.get("EntryMean_h"), default=None)
        mean_h = safe_float(row.get("MeanDelay_h"), default=None)
        if mean_h is None:
            mean_h = max(0.0, safe_float(exit_mean, 0.0)) + max(
                0.0, safe_float(entry_mean, 0.0))
        if mean_h <= 0.0:
            continue
        source_year_raw = safe_float(row.get("SourceYear"), default=None)
        source_year = (
            int(source_year_raw) if source_year_raw is not None else None)
        definition = BorderEventDefinition(
            exit_node=exit_node,
            entry_node=entry_node,
            mode=mode,
            mean_delay_h=float(mean_h),
            exit_mean_h=(None if exit_mean is None else float(exit_mean)),
            entry_mean_h=(None if entry_mean is None else float(entry_mean)),
            source=str(row.get("Source", "")).strip(),
            source_year=source_year,
        )
        definitions[(exit_node, entry_node, mode)] = definition
    if not definitions:
        raise ValueError(f"No usable border events found in {path}")
    print(f"[INFO] Loaded directional border events: {len(definitions)} "
          f"from {path}")
    return definitions


def _regions_differ(arc: Arc) -> bool:
    return bool(
        arc.from_region and arc.to_region
        and arc.from_region != arc.to_region
    )


def resolve_border_event(
    arc: Arc,
    border_delay_map: Dict[Tuple[str, str], float],
    definitions: Optional[
        Dict[Tuple[str, str, str], BorderEventDefinition]
    ] = None,
    mean_source: str = "directional_csv",
) -> Optional[Tuple[Tuple[str, str, str], float]]:
    """Map one network arc to one directional border event.

    Exact BCP-pair data take precedence.  Legacy direct arcs that skip the
    entry checkpoint (for example Khorgos->Almaty) are mapped to the same
    Khorgos->Altynkol event, so all batches share one scenario shock and the
    two checkpoint sides are not charged twice.
    """
    if mean_source not in ("directional_csv", "node_border"):
        raise ValueError(
            "border mean source must be 'directional_csv' or 'node_border'")
    definitions = (BORDER_EVENT_DEFINITIONS
                   if definitions is None else definitions)
    crosses_region = _regions_differ(arc)

    if mean_source == "node_border":
        if not arc.is_border_arc or not crosses_region:
            return None
        mean_h = (
            max(0.0, safe_float(
                border_delay_map.get((arc.from_node, arc.mode)), 0.0))
            + max(0.0, safe_float(
                border_delay_map.get((arc.to_node, arc.mode)), 0.0))
        )
        if mean_h <= 0.0:
            return None
        return ((arc.from_node, arc.to_node, arc.mode), mean_h)

    for event_key, definition in definitions.items():
        exit_node, entry_node, mode = event_key
        if mode != arc.mode:
            continue
        exact_forward = (
            arc.from_node == exit_node and arc.to_node == entry_node)
        exact_reverse = (
            arc.from_node == entry_node and arc.to_node == exit_node)
        alias_forward = crosses_region and (
            arc.from_node == exit_node or arc.to_node == entry_node)
        alias_reverse = crosses_region and (
            arc.from_node == entry_node or arc.to_node == exit_node)
        if exact_forward or alias_forward:
            return event_key, definition.mean_delay_h
        if exact_reverse or alias_reverse:
            reverse_key = (entry_node, exit_node, mode)
            return reverse_key, definition.mean_delay_h

    # The current network also contains these two gauge-change pairs.  Until
    # a directional source table is supplied, combine the two side-specific
    # Node_Border means exactly as Dbar_b=Dbar_exit+Dbar_entry.
    for exit_node, entry_node, mode in WORKBOOK_BORDER_PAIR_FALLBACKS:
        if arc.mode != mode:
            continue
        exact_forward = (
            arc.from_node == exit_node and arc.to_node == entry_node)
        exact_reverse = (
            arc.from_node == entry_node and arc.to_node == exit_node)
        alias_forward = crosses_region and (
            arc.from_node == exit_node or arc.to_node == entry_node)
        alias_reverse = crosses_region and (
            arc.from_node == entry_node or arc.to_node == exit_node)
        mean_h = (
            max(0.0, safe_float(border_delay_map.get((exit_node, mode)), 0.0))
            + max(0.0, safe_float(
                border_delay_map.get((entry_node, mode)), 0.0))
        )
        if mean_h <= 0.0:
            continue
        if exact_forward or alias_forward:
            return (exit_node, entry_node, mode), mean_h
        if exact_reverse or alias_reverse:
            return (entry_node, exit_node, mode), mean_h

    # The source table uses * -> * for the general road-BCP observation.
    wildcard = definitions.get(("*", "*", arc.mode))
    if wildcard is not None and arc.is_border_arc and crosses_region:
        return ((arc.from_node, arc.to_node, arc.mode),
                wildcard.mean_delay_h)
    return None


# ════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════

def load_carbon_tax_map(xls):
    out = {}
    if "Carbon_Tax" not in xls.sheet_names: return out
    try:
        df = pd.read_excel(xls, "Carbon_Tax")
        rc = next((c for c in ["RegionGroup","Region","region","RegionCode"] if c in df.columns), None)
        tc = next((c for c in ["CT_USD_per_tonCO2","CarbonTax_$_per_tCO2","CarbonTax","CT","Tax"] if c in df.columns), None)
        if rc and tc:
            for _, row in df.iterrows():
                r = str(row.get(rc,"")).strip()
                if r: out[r] = safe_float(row.get(tc), default=0.0)
        print(f"[INFO] Loaded carbon tax for {len(out)} regions.")
    except Exception as e:
        print(f"[WARN] Failed to read Carbon_Tax ({e}).")
    return out

def load_carbon_tax_applicability(xls):
    """读取 Carbon_Tax_Applicability sheet → {(region, mode): 0 or 1}"""
    out = {}
    if "Carbon_Tax_Applicability" not in xls.sheet_names:
        print("[WARN] Carbon_Tax_Applicability sheet 不存在，默认全部 mode 征税。")
        return out
    try:
        df = pd.read_excel(xls, "Carbon_Tax_Applicability")
        rc = next((c for c in ["RegionGroup", "Region", "region"] if c in df.columns), None)
        mc = next((c for c in ["Mode", "mode"] if c in df.columns), None)
        ac = next((c for c in ["Applies", "Theta", "applicability"] if c in df.columns), None)
        if rc and mc and ac:
            for _, row in df.iterrows():
                r = str(row.get(rc, "")).strip()
                m = normalize_mode(row.get(mc, ""))
                a = int(safe_float(row.get(ac), default=1.0))
                if r and m:
                    out[(r, m)] = a
        print(f"[INFO] Loaded carbon tax applicability: {len(out)} entries.")
    except Exception as e:
        print(f"[WARN] Failed to load Carbon_Tax_Applicability ({e}).")
    return out

def load_border_delay_map(xls, include_customs: bool = True):
    """[V1 WIRING] Reads PER-MODE border delay columns.

    The expanded data has `BorderDelay_road_h / _rail_h / _water_h`
    (+ `CustomsClearance_h`) and NO single `Mode`/`BorderDelay_h` column,
    so the old single-column loader silently returned {} → delays were
    never applied. This version reads the per-mode columns and falls back
    to the legacy single-column format for backward compatibility.

    Returns {(node, mode): delay_h}. Delay is applied in
    simulate_path_time_capacity ONLY at BREAK_OF_GAUGE_NODES, so extra
    entries here (e.g. for mislabeled EU cities) are harmless.
    """
    out = {}
    if "Node_Border" not in xls.sheet_names: return out
    try:
        nb = pd.read_excel(xls, "Node_Border")
        nc = next((c for c in ["EnglishName","NodeEN"] if c in nb.columns), None)
        if nc is None: return out
        permode = {"road": "BorderDelay_road_h",
                   "rail": "BorderDelay_rail_h",
                   "water": "BorderDelay_water_h"}
        has_permode = any(col in nb.columns for col in permode.values())
        cc_col = "CustomsClearance_h" if "CustomsClearance_h" in nb.columns else None
        if has_permode:
            for _, row in nb.iterrows():
                n = str(row.get(nc, "")).strip()
                if not n: continue
                cc = (
                    safe_float(row.get(cc_col), default=0.0)
                    if include_customs and cc_col else 0.0)
                for m, col in permode.items():
                    if col in nb.columns:
                        d = safe_float(row.get(col), default=0.0)
                        if d > 0.0 or cc > 0.0:
                            out[(n, m)] = d + cc
            suffix = " + customs" if include_customs else " only"
            print(
                f"[INFO] Loaded border delay map (per-mode{suffix}): "
                f"{len(out)} entries.")
        else:
            mc = next((c for c in ["Mode","mode"] if c in nb.columns), None)
            dc = next((c for c in ["BorderDelay_h","Delay_h","BD"] if c in nb.columns), None)
            if nc and mc and dc:
                for _, row in nb.iterrows():
                    n = str(row.get(nc,"")).strip()
                    m = normalize_mode(row.get(mc,""))
                    if n and m:
                        out[(n, m)] = safe_float(row.get(dc), default=0.0)
            print(f"[INFO] Loaded border delay map (legacy): {len(out)} entries.")
    except Exception as e:
        print(f"[WARN] Failed to load border delay ({e}).")
    return out


def load_border_delay_map_from_workbook(
        workbook_path, include_customs: bool = True):
    """Read Node_Border delay means directly from one workbook path."""
    with pd.ExcelFile(workbook_path) as xls:
        return load_border_delay_map(xls, include_customs=include_customs)


def load_border_capacity_map(xls):
    """[V1 WIRING] Reads BorderCapacity_TEUday + BackgroundFlow_TEUday.

    These columns existed in the data but were never read by the model.
    Returns (capacity_map, background_map) keyed by node name. Only
    positive values are stored. The capacity soft-constraint in
    evaluate_individual applies only at BOTTLENECK_NODES with cap>0.
    """
    cap, bg = {}, {}
    if "Node_Border" not in xls.sheet_names: return cap, bg
    try:
        nb = pd.read_excel(xls, "Node_Border")
        nc = next((c for c in ["EnglishName","NodeEN"] if c in nb.columns), None)
        capc = "BorderCapacity_TEUday" if "BorderCapacity_TEUday" in nb.columns else None
        bgc  = "BackgroundFlow_TEUday" if "BackgroundFlow_TEUday" in nb.columns else None
        if nc:
            for _, row in nb.iterrows():
                n = str(row.get(nc, "")).strip()
                if not n: continue
                if capc:
                    v = safe_float(row.get(capc), default=0.0)
                    if v > 0.0: cap[n] = v
                if bgc:
                    v = safe_float(row.get(bgc), default=0.0)
                    if v > 0.0: bg[n] = v
        print(f"[INFO] Loaded border capacity: {len(cap)} nodes | "
              f"background flow: {len(bg)} nodes.")
    except Exception as e:
        print(f"[WARN] Failed to load border capacity ({e}).")
    return cap, bg


def load_emission_factor_map(xls):
    out = {}
    if "Emission_Factors" not in xls.sheet_names: return out
    try:
        df  = pd.read_excel(xls, "Emission_Factors")
        mc  = next((c for c in ["Mode","mode"] if c in df.columns), None)
        rc  = next((c for c in ["RegionGroup","Region","region","RegionCode"] if c in df.columns), None)
        efc = next((c for c in ["gCO2_per_TEU_km_assuming10t","EmissionFactor","Emission_gCO2_per_TEU_km","EF","value"] if c in df.columns), None)
        if mc and rc and efc:
            for _, row in df.iterrows():
                m = normalize_mode(row.get(mc,""))
                r = str(row.get(rc,"")).strip()
                if m and r:
                    factor = safe_float(row.get(efc), default=0.0)
                    if efc == "gCO2_per_TEU_km_assuming10t":
                        factor *= PAYLOAD_TONNES_PER_TEU / 10.0
                    out[(m, r)] = factor
        print(f"[INFO] Loaded emission factors for {len(out)} (mode, region) pairs.")
    except Exception as e:
        print(f"[WARN] Failed to read Emission_Factors ({e}).")
    return out


def load_mode_speeds(xls):
    out = {}
    if "Mode_Speeds" not in xls.sheet_names: return out
    try:
        df  = pd.read_excel(xls, "Mode_Speeds")
        mc  = next((c for c in ["Mode","mode"] if c in df.columns), None)
        spc = next((c for c in ["Speed_kmh","speed_kmh","Speed"] if c in df.columns), None)
        if mc and spc:
            for _, row in df.iterrows():
                m = normalize_mode(row.get(mc,""))
                if m: out[m] = safe_float(row.get(spc), default=0.0)
        print(f"[INFO] Loaded mode speeds: {out}")
    except Exception as e:
        print(f"[WARN] Failed to read Mode_Speeds ({e}).")
    return out


def resolve_effective_speed_map(
    cli_overrides: Dict[str, Optional[float]],
    workbook_speeds: Dict[str, float],
    fallback: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Resolve one nominal mode-speed map: CLI override > workbook > fallback.

    ``cli_overrides[mode]`` is None unless the user explicitly passed the
    corresponding --{mode}-speed-kmh flag. ``workbook_speeds`` is whatever
    load_mode_speeds(xls) returned for this workbook (may omit a mode, or
    hold a non-positive placeholder). ``fallback`` (DEFAULT_MODE_SPEED_KMH)
    is used only when neither higher-priority source supplies a positive
    value for that mode. Returns (effective_speed_map, source_map) where
    source_map[mode] is one of "cli_override" / "workbook_mode_speeds" /
    "hardcoded_fallback", for logging/manifest provenance.
    """
    effective: Dict[str, float] = {}
    source: Dict[str, str] = {}
    for mode, fallback_speed in fallback.items():
        cli_value = cli_overrides.get(mode) if cli_overrides else None
        workbook_value = workbook_speeds.get(mode) if workbook_speeds else None
        if cli_value is not None and cli_value > 0:
            effective[mode] = float(cli_value)
            source[mode] = "cli_override"
        elif workbook_value is not None and workbook_value > 0:
            effective[mode] = float(workbook_value)
            source[mode] = "workbook_mode_speeds"
        else:
            effective[mode] = float(fallback_speed)
            source[mode] = "hardcoded_fallback"
    return effective, source


def positive_speed_kmh_arg(value: str) -> float:
    """argparse ``type=`` for --{mode}-speed-kmh: an explicitly supplied CLI
    speed override must be numeric, finite, and strictly positive. This is
    enforced HERE, at the CLI boundary, so an invalid explicit override is
    a clear, immediate error rather than being silently treated as "no
    override" and falling through to the workbook value -- the user asked
    for the speed they typed, not for it to be second-guessed.

    ``float()`` alone accepts "nan"/"inf"/"-inf" as valid floats, and NaN
    comparisons are always False (`nan <= 0.0` is False), so a bare
    `parsed <= 0.0` check would let NaN and +Infinity silently through --
    NaN would then fail resolve_effective_speed_map()'s own `> 0` check and
    fall through to the workbook (exactly the silent-fallback behaviour
    this validator exists to prevent), while +Infinity would be accepted
    outright as the effective speed. math.isfinite() rejects NaN and both
    infinities explicitly, in addition to the existing <= 0 rejection of
    zero and negative values.

    This is deliberately stricter than resolve_effective_speed_map()'s own
    `cli_value > 0` check, which stays lenient/defensive for callers that
    build the cli_overrides dict programmatically (e.g. tests) rather than
    through argparse; the hard rejection belongs at the user-facing CLI
    layer, not inside the pure resolver.

    Only invoked by argparse when the flag is actually supplied -- an
    omitted flag keeps its ``default=None`` without this function running.
    """
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid number (km/h).")
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"must be a finite positive speed in km/h, got {value!r} "
            "(NaN, +/-Infinity, zero, and negative values are all "
            "rejected). Omit this flag entirely to let the workbook "
            "Mode_Speeds value (or the hardcoded fallback) apply instead "
            "of an explicit override.")
    return parsed


def resolve_stochastic_eval_and_scenarios(
    risk_metric: str, no_stochastic: bool, mc_scenarios_arg: int,
) -> Tuple[bool, int]:
    """Resolve (STOCHASTIC_EVAL, MC_SCENARIOS) from CLI inputs.

    Pulled out of the `if __name__ == "__main__":` block, unchanged, so
    tests can verify --risk-metric deterministic genuinely forces a single
    nominal scenario (stochastic_eval=False, i.e. S=1 once passed to
    configure_scenario_set) via the REAL configuration path, rather than a
    re-implementation of the rule. Behaviour (including the printed
    [INFO]/[WARN] messages) is identical to the code this replaced.
    """
    stochastic_eval = not no_stochastic
    if risk_metric == "deterministic":
        if stochastic_eval:
            print("[INFO] --risk-metric deterministic forces a single "
                  "nominal scenario (S=1); --mc-scenarios is ignored.")
        stochastic_eval = False
    elif risk_metric == "ev":
        if stochastic_eval:
            print("[INFO] --risk-metric ev uses one direct expected-input "
                  "environment (S=1); --mc-scenarios is ignored.")
        stochastic_eval = False
    elif not stochastic_eval:
        print(f"[WARN] --risk-metric {risk_metric} was requested together "
              "with --no-stochastic (S=1). This is a degenerate, "
              "zero-variance run and is equivalent to deterministic mode "
              "for this risk metric.")
    mc_scenarios = max(1, mc_scenarios_arg)
    if risk_metric in ("ev", "ccp") and stochastic_eval and mc_scenarios <= 1:
        print(f"[WARN] --risk-metric {risk_metric} with --mc-scenarios "
              f"{mc_scenarios} produces no sampling variance; consider a "
              "larger scenario count for a meaningful EV/CCP estimate.")
    return stochastic_eval, mc_scenarios


def build_scenario_manifest(
    *,
    risk_metric: str,
    scenario_set: "ScenarioSet",
    confidence_cost: float,
    confidence_emission: float,
    confidence_time: float,
    mode_time_cv: Dict[str, float],
    mode_time_cap_factor: Dict[str, float],
    mode_speed_kmh: Dict[str, float],
    mode_speed_source: Dict[str, str],
    mode_speed_requested_cli_override_kmh: Dict[str, Optional[float]],
    border_delay_cv: Dict[str, float],
    border_delay_cap_factor: float,
    border_event_data_file: str,
    border_event_definitions: Dict[Tuple[str, str, str], "BorderEventDefinition"],
    late_penalty_source: str,
    late_penalty_input_basis: str,
    late_penalty_baseline_usd_per_teu_h: float,
    use_input_late_penalties: bool,
    payload_tonnes_per_teu: float,
    wait_emission_g_per_teu_h: float,
    feasibility_seed_fraction: float,
    feasibility_search_restarts: int,
    feasibility_search_iterations: int,
) -> Dict[str, Any]:
    """Build the scenario_manifest.json payload.

    Pulled out of the `if __name__ == "__main__":` block, unchanged field
    for field, so tests can construct/inspect the REAL manifest logic per
    mode (e.g. confirm "fitness_evaluation" and "mode_speed_kmh" are
    mode-correct, confirm "risk_metric"/"scenario_count"/"scenario_seed"
    are recorded correctly) without running the full script. Every key and
    value below is identical to the inline dict this replaced.
    """
    return {
        "risk_metric": risk_metric,
        "scenario_count": scenario_set.size,
        "scenario_seed": scenario_set.seed,
        "stochastic": scenario_set.stochastic,
        "confidence": {
            "cost": confidence_cost,
            "emission": confidence_emission,
            "time": confidence_time,
        },
        "mode_time_cv": mode_time_cv,
        "mode_time_cap_factor": mode_time_cap_factor,
        # Effective mode speed actually applied to every Arc this run
        # (CLI override > workbook Mode_Speeds > hardcoded fallback), NOT
        # the raw fallback/default constants -- see mode_speed_source for
        # per-mode provenance.
        "mode_speed_kmh": mode_speed_kmh,
        "mode_speed_source": mode_speed_source,
        "mode_speed_requested_cli_override_kmh":
            mode_speed_requested_cli_override_kmh,
        "timetable_role": (
            "frequency/first_departure/headway determine schedule waiting; "
            "arc mean running time is distance/mode_speed"),
        "border_delay_cv": border_delay_cv,
        "border_delay_cap_factor": border_delay_cap_factor,
        "border_event_data_file": border_event_data_file,
        "border_event_source_records": [
            {
                "event": "|".join(key),
                "mean_delay_h": definition.mean_delay_h,
                "exit_mean_h": definition.exit_mean_h,
                "entry_mean_h": definition.entry_mean_h,
                "source": definition.source,
                "source_year": definition.source_year,
            }
            for key, definition in border_event_definitions.items()
        ],
        "border_event_fallback": (
            "Manzhouli-Zabaykalsk and Brest-Malaszewicze use the sum "
            "of available Node_Border side means until directional "
            "records are added to the source CSV"),
        "border_event_means_h": {
            "|".join(key): value
            for key, value in scenario_set.border_event_mean_h.items()
        },
        "late_penalty": {
            "source_mode": late_penalty_source,
            "input_basis": late_penalty_input_basis,
            "baseline_usd_per_teu_day":
                late_penalty_baseline_usd_per_teu_h * 24.0,
            "applied_common_usd_per_teu_hour": (
                None if use_input_late_penalties
                else late_penalty_baseline_usd_per_teu_h),
        },
        "payload_tonnes_per_teu": payload_tonnes_per_teu,
        "wait_emission_g_per_teu_h": wait_emission_g_per_teu_h,
        **waiting_emission_configuration(wait_emission_g_per_teu_h),
        "fitness_evaluation": FITNESS_EVALUATION_DESCRIPTION.get(
            risk_metric, f"unknown_risk_metric:{risk_metric}"),
        "algorithmic_randomness": "controlled separately by --seed",
        "feasibility_first_search": {
            "seed_fraction": feasibility_seed_fraction,
            "restarts": feasibility_search_restarts,
            "iterations_per_restart": feasibility_search_iterations,
            "single_path_reliability_screen": True,
            "capacity_aware_min_conflicts": True,
            "constraint_selection": "normalized_Deb_constraint_domination",
        },
    }


def load_transshipment_map(xls):
    out = {}
    if "Transshipment" not in xls.sheet_names: return out
    try:
        df   = pd.read_excel(xls, "Transshipment")
        ndc  = next((c for c in ["Node","NodeEN","EnglishName"] if c in df.columns), None)
        imc  = next((c for c in ["InMode","FromMode","mode_in"] if c in df.columns), None)
        omc  = next((c for c in ["OutMode","ToMode","mode_out"] if c in df.columns), None)
        cstc = next((c for c in [
            "TransferCost_USD_per_TEU", "TransCost", "Cost",
            "trans_cost", "Cost_per_TEU"] if c in df.columns), None)
        tmc  = next((c for c in [
            "TransferTime_h", "TransTime_h", "Time_h",
            "trans_time_h", "Time"] if c in df.columns), None)
        if ndc and imc and omc:
            for _, row in df.iterrows():
                node     = str(row.get(ndc,"")).strip()
                in_mode  = normalize_mode(row.get(imc,""))
                out_mode = normalize_mode(row.get(omc,""))
                if node and in_mode and out_mode:
                    out[(node, in_mode, out_mode)] = {
                        "cost_per_teu": safe_float(row.get(cstc), default=0.0) if cstc else 0.0,
                        "time_h":       safe_float(row.get(tmc),  default=0.0) if tmc  else 0.0,
                    }
        print(f"[INFO] Loaded transshipment entries: {len(out)}")
    except Exception as e:
        print(f"[WARN] Failed to read Transshipment ({e}).")
    return out


def load_waiting_params(xls):
    wc = WAITING_COST_PER_TEU_HOUR_DEFAULT
    we = WAIT_EMISSION_gCO2_per_TEU_H_DEFAULT
    if "Waiting_Costs" not in xls.sheet_names: return wc, we
    try:
        df = pd.read_excel(xls, "Waiting_Costs")
        def pick(colnames, default):
            for c in colnames:
                if c in df.columns:
                    vals = df[c].dropna().tolist()
                    if vals: return safe_float(vals[0], default=default)
            return default
        wc = pick(["W_hold_USD_per_TEU_h", "WaitingCost_per_TEU_h",
                   "WaitCost_per_TEU_h"], wc)
        we = pick(["WaitEmission_gCO2_per_TEU_h","WaitingEmission_gCO2_per_TEU_h"], we)
        print(f"[INFO] Loaded waiting params: cost={wc}, emission={we}")
    except Exception as e:
        print(f"[WARN] Failed to read Waiting_Costs ({e}).")
    return wc, we


def load_network_from_extended(
    filename: str,
    cli_speed_override: Optional[Dict[str, Optional[float]]] = None,
):
    global CHINA_BORDER_NODES, NODE_GROUP, BORDER_CAPACITY, BACKGROUND_FLOW
    global EFFECTIVE_MODE_SPEED_KMH, EFFECTIVE_MODE_SPEED_SOURCE
    xls = pd.ExcelFile(filename)

    carbon_tax_map      = load_carbon_tax_map(xls)
    emission_factor_map = load_emission_factor_map(xls)
    input_mode_speeds_map = load_mode_speeds(xls)
    if cli_speed_override is None:
        cli_speed_override = CLI_MODE_SPEED_OVERRIDE_KMH
    mode_speeds_map, mode_speed_source = resolve_effective_speed_map(
        cli_speed_override, input_mode_speeds_map, DEFAULT_MODE_SPEED_KMH)
    EFFECTIVE_MODE_SPEED_KMH   = dict(mode_speeds_map)
    EFFECTIVE_MODE_SPEED_SOURCE = dict(mode_speed_source)
    print("[INFO] Effective mode speeds (CLI override > workbook Mode_Speeds "
          f"> hardcoded fallback): { {m: (mode_speeds_map[m], mode_speed_source[m]) for m in sorted(mode_speeds_map)} }")
    trans_map           = load_transshipment_map(xls)
    border_delay_map    = load_border_delay_map(xls)
    theta_rm = load_carbon_tax_applicability(xls)

    # [V1 WIRING] populate node capacity & background-flow globals
    BORDER_CAPACITY, BACKGROUND_FLOW = load_border_capacity_map(xls)
    _active_caps = {n: BORDER_CAPACITY[n] for n in BREAK_OF_GAUGE_NODES
                    if n in BORDER_CAPACITY}
    print(f"[INFO] Active break-of-gauge capacities: {_active_caps}")
    print(f"[INFO] Active break-of-gauge background flow: "
          f"{ {n: BACKGROUND_FLOW.get(n,0.0) for n in BREAK_OF_GAUGE_NODES if n in BACKGROUND_FLOW} }")

    nodes_df   = pd.read_excel(xls, "Nodes")
    node_names = nodes_df["EnglishName"].astype(str).str.strip().tolist()

    node_region = {
        str(name).strip(): norm_region(reg)
        for name, reg in zip(nodes_df["EnglishName"], nodes_df["Region"])
    }

    if "RegionGroup" in nodes_df.columns:
        NODE_GROUP = {
            str(name).strip(): str(grp).strip()
            for name, grp in zip(nodes_df["EnglishName"], nodes_df["RegionGroup"])
            if str(grp).strip() not in ("", "nan", "None")
        }

    if "Node_Border" in xls.sheet_names:
        try:
            nb_df = pd.read_excel(xls, "Node_Border")
            loaded_borders = set()
            for _, row in nb_df.iterrows():
                region_val = str(row.get("Region","")).strip()
                is_border  = safe_float(row.get("IsBorderNode", 0), default=0.0) == 1.0
                if region_val == "CN" and is_border:
                    node_name = str(row.get("EnglishName","")).strip()
                    if node_name: loaded_borders.add(node_name)
            if loaded_borders:
                CHINA_BORDER_NODES = loaded_borders
                print(f"[INFO] Loaded China border nodes ({len(CHINA_BORDER_NODES)}): {sorted(CHINA_BORDER_NODES)}")
        except Exception as e:
            print(f"[WARN] Failed to load Node_Border: {e}. Using defaults.")
    else:
        print(f"[INFO] Node_Border sheet not found. Using defaults: {sorted(CHINA_BORDER_NODES)}")

    CHINA_BORDER_NODES.update({"Ningbo", "Shanghai"})

    node_hold_cost:  Dict[str, float] = {}
    node_proc_cost:  Dict[str, float] = {}
    node_trans_cost: Dict[str, float] = {}   # 新增：W_j^trans
    for _, row in nodes_df.iterrows():
        n = str(row.get("EnglishName","")).strip()
        node_hold_cost[n]  = safe_float(row.get("HoldCost_per_TEU_h"),
                                        default=WAITING_COST_PER_TEU_HOUR_DEFAULT)
        node_proc_cost[n]  = safe_float(row.get("ProcCost_per_TEU_h"),
                                        default=0.0)
        node_trans_cost[n] = safe_float(row.get("TransOpCost_per_TEU_h"),
                                        default=node_proc_cost[n])

    SEAPORT_NODES = {"Ningbo", "Shanghai"}
    for n in SEAPORT_NODES:
        if n not in node_region:
            node_region[n] = "CN"
            NODE_GROUP[n]  = "CN"
            if n not in node_names: node_names.append(n)

    waiting_cost_per_teu_h, wait_emis_g_per_teu_h = load_waiting_params(xls)

    arcs_df   = pd.read_excel(xls, "Arcs_All")
    arcs: List[Arc] = []
    cost_cols = ["Cost_$_per_km","Cost_per_km","Cost"]
    emis_cols = ["Emission_gCO2_per_tkm","Emission_gCO2_per_TEU_km","EmissionFactor","Emission"]

    for _, row in arcs_df.iterrows():
        mode  = normalize_mode(row.get("Mode","road"))
        speed = {"road": 75.0, "water": 30.0}.get(mode, 50.0)
        if mode in mode_speeds_map and mode_speeds_map[mode] > 0:
            speed = mode_speeds_map[mode]

        origin = str(row.get("OriginEN","")).strip()
        dest   = str(row.get("DestEN","")).strip()
        if _is_bad_text_token(origin) or _is_bad_text_token(dest): continue

        from_region = str(node_region.get(origin,"")).strip()
        to_region   = str(node_region.get(dest,"")).strip()
        distance    = parse_distance_km(row.get("Distance_km", 0.0))

        if "Capacity_TEUday" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUday", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUday"), default=1e9)
        elif "Capacity_TEUh" in arcs_df.columns and not pd.isna(row.get("Capacity_TEUh", np.nan)):
            capacity = safe_float(row.get("Capacity_TEUh"), default=1e9) * 24.0
        else:
            capacity = 1e9

        cpkm = 0.0
        for c in cost_cols:
            if c in arcs_df.columns:
                val = safe_float(row.get(c), default=None)
                if val is not None and val > 0: cpkm = val; break
        if cpkm <= 1e-9: cpkm = 0.5

        epkm = 0.0
        for c in emis_cols:
            if c in arcs_df.columns:
                epkm = safe_float(row.get(c), default=0.0); break
        if "Emission_gCO2_per_tkm" in arcs_df.columns and epkm > 0:
            epkm = epkm * PAYLOAD_TONNES_PER_TEU
        if (mode, from_region) in emission_factor_map:
            epkm = emission_factor_map[(mode, from_region)]

        arcs.append(Arc(
            from_node=origin, to_node=dest, mode=mode,
            distance=distance, capacity=capacity,
            cost_per_teu_km=cpkm, emission_per_teu_km=epkm,
            speed_kmh=speed, from_region=from_region, to_region=to_region,
            is_border_arc=bool(int(safe_float(
                row.get("IsBorderArc"), default=0.0)))
        ))

    tdf = pd.read_excel(xls, "Timetable")
    timetables: List[TimetableEntry] = []
    travel_time_col = next((
        c for c in ["ScheduledTravelTime_h", "TravelTime_h", "Time_h"]
        if c in tdf.columns
    ), None)
    legacy_time_col = "time" if "time" in tdf.columns else None
    for _, row in tdf.iterrows():
        origin    = str(row.get("OriginEN","")).strip()
        dest      = str(row.get("DestEN","")).strip()
        mode_norm = normalize_mode(row.get("Mode",""))
        if _is_bad_text_token(origin) or _is_bad_text_token(dest): continue
        if mode_norm not in {"road","rail","water"}: continue
        freq   = safe_float(row.get("Frequency_per_week"), default=1.0)
        hd_raw = row.get("Headway_Hours", np.nan)
        hd     = 168.0 / max(freq, 1.0) if pd.isna(hd_raw) else safe_float(hd_raw, default=168.0)
        v      = row.get("FirstDepartureHour", np.nan)
        fd     = 0.0
        if not pd.isna(v):
            try:
                s  = str(v).strip()
                fd = float(s.split(":")[0]) if ":" in s else float(s)
            except Exception:
                fd = 0.0
        timetables.append(TimetableEntry(
            from_node=origin, to_node=dest, mode=mode_norm,
            frequency_per_week=freq, first_departure_hour=fd, headway_hours=hd,
            travel_time_h=safe_float(
                row.get(travel_time_col) if travel_time_col else None,
                default=0.0,
            ) or None,
            legacy_time_value=safe_float(
                row.get(legacy_time_col) if legacy_time_col else None,
                default=0.0,
            ) or None,
        ))
    print("[INFO] Timetable loaded: "
          f"{len(timetables)} scheduled services; using frequency, "
          "first departure and headway for induced waiting.")
    if legacy_time_col:
        legacy_values = sorted({
            float(entry.legacy_time_value)
            for entry in timetables
            if entry.legacy_time_value is not None
        })
        print("[WARN] Timetable.time is loaded only as legacy metadata: its "
              f"unit is undocumented (observed values={legacy_values}); it is "
              "not used as arc travel hours. Running time follows "
              "distance / documented mode speed.")

    bdf     = pd.read_excel(xls, "Batches")
    batches: List[Batch] = []
    for _, row in bdf.iterrows():
        origin = str(row.get("OriginEN","")).strip()
        dest   = str(row.get("DestEN","")).strip()
        if node_region.get(origin) in CHINA_REGIONS and node_region.get(dest) in EUROPE_REGIONS:
            batches.append(Batch(
                batch_id=int(row.get("BatchID", 0)),
                origin=origin, destination=dest,
                quantity=safe_float(row.get("QuantityTEU"), default=0.0),
                ET=safe_float(row.get("ET"), default=0.0),
                LT=safe_float(row.get("LT"), default=0.0),
                penalty_per_teu_h=safe_float(row.get("PenaltyCost_per_TEU_h"),
                                             default=DEFAULT_PENALTY_PER_TEU_H),
                max_late_h=(safe_float(row.get("MaxLate_h"), default=0.0) or None),
            ))

    print(f"[INFO] Batches loaded: {len(batches)}")
    return (
        node_names, node_region,
        node_hold_cost, node_proc_cost, node_trans_cost,
        arcs, timetables, batches,
        waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
        carbon_tax_map, emission_factor_map, mode_speeds_map,
        trans_map, border_delay_map, theta_rm, mode_speed_source
    )


def build_graph(arcs):
    g = {}
    for a in arcs:
        g.setdefault(a.from_node, []).append((a.to_node, a))
    return g


def build_timetable_dict(timetables):
    tt = {}
    for t in timetables:
        tt.setdefault((t.from_node, t.to_node, t.mode), []).append(t)
    return tt


def build_arc_lookup(arcs):
    mp = {}
    for a in arcs:
        k = (a.from_node, a.to_node, a.mode)
        if k not in mp: mp[k] = a
    return mp


def nominal_arc_travel_time(arc: Arc, tt_dict: Dict) -> float:
    """Mean line-haul time; timetables control departures, not arc speed."""
    return float(arc.distance / max(arc.speed_kmh, 1.0))


def build_scenario_set(
    arcs: List[Arc],
    border_delay_map: Dict[Tuple[str, str], float],
    size: int,
    seed: int,
    stochastic: bool = True,
    border_event_definitions: Optional[
        Dict[Tuple[str, str, str], BorderEventDefinition]
    ] = None,
    mode_time_cv: Optional[Dict[str, float]] = None,
    mode_time_max_factor: Optional[Dict[str, Optional[float]]] = None,
    border_delay_cv: Optional[Dict[str, float]] = None,
    border_delay_max_factor: Optional[float] = None,
    border_mean_source: str = "directional_csv",
) -> ScenarioSet:
    """Generate the complete finite scenario set once, before optimisation."""
    size = max(1, int(size)) if stochastic else 1
    rng = np.random.default_rng(int(seed))
    effective_mode_cv = MODE_TIME_CV if mode_time_cv is None else mode_time_cv
    effective_mode_cap = (
        MODE_TIME_MAX_FACTOR
        if mode_time_max_factor is None else mode_time_max_factor)
    effective_border_cv = (
        BORDER_DELAY_CV if border_delay_cv is None else border_delay_cv)
    effective_border_cap = (
        BORDER_DELAY_MAX_FACTOR
        if border_delay_max_factor is None else float(border_delay_max_factor))
    travel_multiplier: Dict[Tuple[str, str, str], np.ndarray] = {}
    unique_arc_keys = sorted({(a.from_node, a.to_node, a.mode) for a in arcs})
    for from_node, to_node, mode in unique_arc_keys:
        if stochastic:
            travel_multiplier[(from_node, to_node, mode)] = \
                sample_capped_mean_one_lognormal(
                    size=size,
                    cv=effective_mode_cv.get(mode, 0.10),
                    cap_factor=effective_mode_cap.get(mode),
                    rng=rng,
                )
        else:
            travel_multiplier[(from_node, to_node, mode)] = np.ones(1, dtype=float)

    definitions = (BORDER_EVENT_DEFINITIONS
                   if border_event_definitions is None
                   else border_event_definitions)
    arc_border_event: Dict[
        Tuple[str, str, str], Tuple[str, str, str]
    ] = {}
    border_event_mean_h: Dict[Tuple[str, str, str], float] = {}
    for arc in arcs:
        resolved = resolve_border_event(
            arc, border_delay_map, definitions=definitions,
            mean_source=border_mean_source)
        if resolved is None:
            continue
        event_key, mean_h = resolved
        arc_border_event[(arc.from_node, arc.to_node, arc.mode)] = event_key
        previous = border_event_mean_h.get(event_key)
        if previous is not None and abs(previous - mean_h) > 1e-9:
            raise ValueError(
                f"Conflicting means for border event {event_key}: "
                f"{previous} vs {mean_h}")
        border_event_mean_h[event_key] = float(mean_h)

    border_delay_h: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key in sorted(border_event_mean_h):
        mode = key[2]
        base_h = max(0.0, safe_float(border_event_mean_h[key], 0.0))
        if mode == "water" and base_h > 0.0 and stochastic:
            # No empirical maritime/port border-delay MEAN exists anywhere
            # in this repository as of this framework's design (see
            # BORDER_DELAY_CV comment above). A nonzero water event mean
            # means new data was supplied since then -- surface that
            # explicitly rather than silently starting to sample it. Only
            # relevant when stochastic sampling is actually enabled: under
            # stochastic=False the mean is applied as a constant, not
            # sampled, so no "sampling is now active" claim would be true.
            print(f"[WARN] Nonzero maritime border-delay mean detected for "
                  f"event {key} (mean_h={base_h:g}). Stochastic maritime "
                  f"border-delay sampling "
                  f"(CV={effective_border_cv.get(mode, 0.50):g}) "
                  "will now be ACTIVE for this event. Confirm this mean is "
                  "evidence-based before trusting results that depend on it.")
        if base_h <= 0.0:
            border_delay_h[key] = np.zeros(size, dtype=float)
        elif stochastic:
            multipliers = sample_capped_mean_one_lognormal(
                size=size,
                cv=effective_border_cv.get(mode, 0.50),
                cap_factor=effective_border_cap,
                rng=rng,
            )
            border_delay_h[key] = base_h * multipliers
        else:
            border_delay_h[key] = np.full(1, base_h, dtype=float)

    return ScenarioSet(
        size=size,
        seed=int(seed),
        travel_multiplier=travel_multiplier,
        border_delay_h=border_delay_h,
        arc_border_event=arc_border_event,
        border_event_mean_h=border_event_mean_h,
        stochastic=bool(stochastic),
    )


def build_expected_value_scenario_set(
    arcs: List[Arc],
    border_delay_map: Dict[Tuple[str, str], float],
    seed: int = 0,
    border_event_definitions: Optional[
        Dict[Tuple[str, str, str], BorderEventDefinition]
    ] = None,
    border_mean_source: str = "directional_csv",
) -> ScenarioSet:
    """Build the direct expected-input environment used by EV optimisation.

    This deliberately contains one constructed environment, not a Monte Carlo
    draw and not the mean of a CCP training sample.  The current uncertainty
    model has mean-one line-haul multipliers and mean border-event delays, so
    those expectations are inserted directly.  ``seed`` is provenance only;
    no random generator is consulted.

    The values may coincide numerically with the nominal deterministic case,
    but callers must retain the distinct ``ev`` method label.
    """
    expected = build_scenario_set(
        arcs=arcs,
        border_delay_map=border_delay_map,
        size=1,
        seed=int(seed),
        stochastic=False,
        border_event_definitions=border_event_definitions,
        border_mean_source=border_mean_source,
    )
    return ScenarioSet(
        size=expected.size,
        seed=int(seed),
        travel_multiplier=expected.travel_multiplier,
        border_delay_h=expected.border_delay_h,
        arc_border_event=expected.arc_border_event,
        border_event_mean_h=expected.border_event_mean_h,
        stochastic=False,
    )


def configure_scenario_set(
    arcs: List[Arc],
    border_delay_map: Dict[Tuple[str, str], float],
    size: int,
    seed: int,
    stochastic: bool = True,
    border_event_definitions: Optional[
        Dict[Tuple[str, str, str], BorderEventDefinition]
    ] = None,
) -> ScenarioSet:
    global ACTIVE_SCENARIO_SET, _PATH_SCENARIO_CACHE
    ACTIVE_SCENARIO_SET = build_scenario_set(
        arcs=arcs,
        border_delay_map=border_delay_map,
        size=size,
        seed=seed,
        stochastic=stochastic,
        border_event_definitions=border_event_definitions,
    )
    _PATH_SCENARIO_CACHE = {}
    return ACTIVE_SCENARIO_SET


def load_batches_from_csv(filename: str) -> List[Batch]:
    df = pd.read_csv(filename)
    columns = {
        "batch_id": next((c for c in ["batch_id", "BatchID"] if c in df.columns), None),
        "origin": next((c for c in ["origin", "OriginEN"] if c in df.columns), None),
        "destination": next((c for c in ["destination", "DestEN"] if c in df.columns), None),
        "quantity": next((c for c in ["quantity", "QuantityTEU"] if c in df.columns), None),
        "ET": "ET" if "ET" in df.columns else None,
        "LT": "LT" if "LT" in df.columns else None,
    }
    missing = [name for name, column in columns.items() if column is None]
    if missing:
        raise ValueError(f"Batch CSV missing required fields: {missing}")

    batches: List[Batch] = []
    for _, row in df.iterrows():
        batches.append(Batch(
            batch_id=int(row.get(columns["batch_id"], 0)),
            origin=str(row.get(columns["origin"], "")).strip(),
            destination=str(row.get(columns["destination"], "")).strip(),
            quantity=safe_float(row.get(columns["quantity"]), default=0.0),
            ET=safe_float(row.get(columns["ET"]), default=0.0),
            LT=safe_float(row.get(columns["LT"]), default=0.0),
            penalty_per_teu_h=safe_float(
                row.get("penalty_per_teu_h",
                        row.get("PenaltyCost_per_TEU_h")),
                default=DEFAULT_PENALTY_PER_TEU_H,
            ),
            max_late_h=(safe_float(
                row.get("max_late_h", row.get("MaxLate_h")),
                default=0.0) or None),
        ))
    print(f"[INFO] Batches overridden from CSV: {filename} ({len(batches)} batches)")
    return batches


# ════════════════════════════════════════════════════════
# Path library
# ════════════════════════════════════════════════════════

def random_dfs_paths(graph, origin, dest, node_region,
                     max_len=12, max_paths=200, timeout_sec=8.0):
    deadline = time.time() + timeout_sec
    paths, found_set = [], set()
    attempts, max_attempts = 0, max_paths * 20
    while len(paths) < max_paths and attempts < max_attempts:
        if time.time() > deadline: break
        attempts += 1
        node, cur_arcs, visited, cur_nodes, ok = origin, [], {origin}, [origin], True
        for _ in range(max_len):
            if node == dest: break
            neighbors = list(graph.get(node, []))
            if not neighbors: ok = False; break
            random.shuffle(neighbors)
            moved = False
            for nxt, arc in neighbors:
                if nxt in visited: continue
                new_nodes = cur_nodes + [nxt]
                if not china_border_monotone_ok(new_nodes, node_region): continue
                if not region_monotone_ok(new_nodes, node_region): continue
                cur_arcs.append(arc); visited.add(nxt)
                cur_nodes.append(nxt); node = nxt; moved = True; break
            if not moved: ok = False; break
        if ok and node == dest and cur_arcs:
            key = (tuple(cur_nodes), tuple(a.mode for a in cur_arcs))
            if key not in found_set:
                found_set.add(key); paths.append(cur_arcs)
    return paths


def repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup):
    new_seq = []
    for arc in arc_seq:
        if arc.mode == "road": new_seq.append(arc); continue
        if tt_dict.get((arc.from_node, arc.to_node, arc.mode), []):
            new_seq.append(arc); continue
        k_road = (arc.from_node, arc.to_node, "road")
        if k_road in arc_lookup: new_seq.append(arc_lookup[k_road])
        else: return None
    return new_seq


def select_topk_by_cost_time_emis(paths, k=30, cap_total=90):
    if not paths: return []
    by_cost = sorted(paths, key=lambda p: p.base_cost_per_teu)
    by_time = sorted(paths, key=lambda p: p.base_travel_time_h)
    by_emis = sorted(paths, key=lambda p: p.base_emission_per_teu)
    picked, used = [], set()
    for lst in [by_cost, by_time, by_emis]:
        for p in lst[:k]:
            if p not in used: picked.append(p); used.add(p)
    return picked[:cap_total] if cap_total else picked


def build_path_library(node_names, node_region, arcs, batches, tt_dict, arc_lookup):
    graph    = build_graph(arcs)
    path_lib = {}
    next_pid = 0
    for b in batches:
        od = (b.origin, b.destination)
        if od in path_lib: continue
        arc_paths = random_dfs_paths(graph, b.origin, b.destination,
                                     node_region=node_region, max_len=12,
                                     max_paths=DFS_MAX_PATHS_PER_OD)
        paths_od = []
        for arc_seq in arc_paths:
            repaired = repair_arc_seq_with_road_fallback(arc_seq, tt_dict, arc_lookup)
            if repaired is None: continue
            nodes = [repaired[0].from_node] + [a.to_node for a in repaired]
            if len(set(nodes)) != len(nodes): continue
            if not region_monotone_ok(nodes, node_region): continue
            if not china_border_monotone_ok(nodes, node_region): continue
            modes = [a.mode for a in repaired]
            paths_od.append(Path(
                path_id=next_pid, origin=b.origin, destination=b.destination,
                nodes=nodes, modes=modes, arcs=repaired,
                base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in repaired),
                base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in repaired),
                base_travel_time_h=sum(nominal_arc_travel_time(a, tt_dict)
                                       for a in repaired),
            ))
            next_pid += 1
        if paths_od:
            path_lib[od] = select_topk_by_cost_time_emis(
                paths_od, k=PATHS_TOPK_PER_CRITERION, cap_total=PATH_LIB_CAP_TOTAL)

    removed = 0
    for od in list(path_lib.keys()):
        before = len(path_lib[od])
        path_lib[od] = [p for p in path_lib[od]
                        if china_border_monotone_ok(p.nodes, node_region)]
        removed += before - len(path_lib[od])
        if not path_lib[od]: del path_lib[od]
    if removed: print(f"[WARN] Post-filter removed {removed} paths.")
    else:       print("[INFO] All paths pass border monotonicity. ✅")
    return path_lib


def sanity_check_path_lib(batches, path_lib):
    missing = [(b.batch_id, (b.origin, b.destination))
               for b in batches if not path_lib.get((b.origin, b.destination), [])]
    if missing:
        for bid, od in missing[:20]:
            print(f"[SANITY] ❌ missing paths Batch {bid} OD={od}")
        raise RuntimeError("Path library missing some ODs.")
    print("[SANITY] ✅ All batches have paths.")


def repair_missing_allocations(ind, batches, path_lib):
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        if ind.od_allocations.get(key, []): continue
        paths = path_lib.get((b.origin, b.destination), [])
        if paths:
            ind.od_allocations[key] = [PathAllocation(path=paths[0], share=1.0)]


# ════════════════════════════════════════════════════════
# Simulation & evaluation
# ════════════════════════════════════════════════════════

def next_departure_time_programB(t: float, entries: List[TimetableEntry]) -> float:
    best = float("inf")
    for e in entries:
        if t <= e.first_departure_hour:
            dep = e.first_departure_hour
        else:
            waited = t - e.first_departure_hour
            n      = math.ceil(waited / max(e.headway_hours, 1e-6))
            dep    = e.first_departure_hour + n * e.headway_hours
        if dep < best: best = dep
    return best if best < float("inf") else t


def simulate_path_time_capacity(
    path: Path, batch: Batch, flow_teu: float,
    tt_dict: Dict, arc_flow_map: Dict,
    trans_map: Optional[Dict] = None,
    border_delay_map: Optional[Dict] = None,
    node_flow_map: Optional[Dict] = None,   # [V1 WIRING] per-(node,day) throughput
    rng: Optional[np.random.Generator] = None,
    stochastic: bool = False,
    mode_time_cv: Optional[Dict[str, float]] = None,
    scenario_index: int = 0,
    record_capacity: bool = True,
) -> Tuple[float, List[Tuple[str, float, float]], int]:
    t                = float(batch.ET)
    miss_tt          = 0
    trans_map        = trans_map or {}
    border_delay_map = border_delay_map or {}
    mode_time_cv     = mode_time_cv or MODE_TIME_CV
    prev_arc         = None
    node_wait_list: List[Tuple[str, float, float]] = []
    incurred_border_events: set = set()

    for arc in path.arcs:
        cur_node       = arc.from_node
        arc_trans_wait = 0.0

        if prev_arc is not None and prev_arc.mode != arc.mode:
            rec = trans_map.get((cur_node, prev_arc.mode, arc.mode))
            if rec:
                th = safe_float(rec.get("time_h"), default=0.0)
                if th > 0: t += th; arc_trans_wait += th

        # Border uncertainty is sampled per directional crossing event
        # b=(exit checkpoint, entry checkpoint, mode), not independently at
        # its two endpoint nodes.  Alias arcs that skip a checkpoint share the
        # same event and the same scenario realisation.
        event_key = None
        bd = 0.0
        if ACTIVE_SCENARIO_SET is not None:
            event_key = ACTIVE_SCENARIO_SET.border_event_for_arc(arc)
            if event_key is not None and event_key not in incurred_border_events:
                if stochastic:
                    _, bd = ACTIVE_SCENARIO_SET.border_for_arc(
                        arc, scenario_index)
                else:
                    bd = ACTIVE_SCENARIO_SET.border_event_mean_h.get(
                        event_key, 0.0)
        else:
            resolved = resolve_border_event(arc, border_delay_map)
            if resolved is not None:
                event_key, bd = resolved
        if event_key is not None and event_key not in incurred_border_events:
            incurred_border_events.add(event_key)
            if bd > 0.0:
                t += bd
                arc_trans_wait += bd

        travel_arc = nominal_arc_travel_time(arc, tt_dict)
        if stochastic and ACTIVE_SCENARIO_SET is not None:
            travel_arc *= ACTIVE_SCENARIO_SET.travel(arc, scenario_index)
        entries    = [] if arc.mode == "road" else \
                     tt_dict.get((cur_node, arc.to_node, arc.mode), [])
        if arc.mode != "road" and not entries:
            miss_tt += 1; return float("inf"), [], miss_tt

        dep            = t if not entries else next_departure_time_programB(t, entries)
        arc_sched_wait = max(0.0, dep - t)
        node_wait_list.append((cur_node, arc_sched_wait, arc_trans_wait))

        arr  = dep + travel_arc
        slot = int(dep // 24)
        akey = (cur_node, arc.to_node, arc.mode)
        if record_capacity:
            arc_flow_map[(akey, slot)] = arc_flow_map.get((akey, slot), 0.0) + flow_teu

        # [V1 WIRING] accumulate departing throughput at bottleneck nodes
        # for per-day node capacity check (counted once per transit).
        if record_capacity and node_flow_map is not None and cur_node in BOTTLENECK_NODES:
            node_flow_map[(cur_node, slot)] = \
                node_flow_map.get((cur_node, slot), 0.0) + flow_teu

        t        = arr
        prev_arc = arc

    return (t - batch.ET), node_wait_list, miss_tt


def simulate_path_over_scenarios(
    path: Path,
    batch: Batch,
    tt_dict: Dict,
    trans_map: Dict,
    border_delay_map: Dict,
    scenario_set: ScenarioSet,
) -> PathScenarioResult:
    """Evaluate one path against an already generated, immutable scenario set."""
    trans_signature = tuple(
        (path.arcs[i + 1].from_node, path.arcs[i].mode,
         path.arcs[i + 1].mode,
         safe_float(trans_map.get((path.arcs[i + 1].from_node,
                                   path.arcs[i].mode,
                                   path.arcs[i + 1].mode), {}).get("time_h"), 0.0))
        for i in range(len(path.arcs) - 1)
        if path.arcs[i].mode != path.arcs[i + 1].mode
    )
    topology = tuple((a.from_node, a.to_node, a.mode) for a in path.arcs)
    cache_key = (
        scenario_set.seed, scenario_set.size, scenario_set.stochastic,
        batch.batch_id, float(batch.ET), topology, trans_signature,
    )
    cached = _PATH_SCENARIO_CACHE.get(cache_key)
    if cached is not None:
        return cached

    size = scenario_set.size
    arrivals = np.full(size, float("inf"), dtype=float)
    schedule_wait_h: Dict[str, np.ndarray] = {}
    border_delay_h: Dict[str, np.ndarray] = {}
    missing_timetable = False

    for s in range(size):
        t = float(batch.ET)
        prev_arc = None
        valid = True
        incurred_border_events: set = set()
        for arc in path.arcs:
            node = arc.from_node
            if prev_arc is not None and prev_arc.mode != arc.mode:
                rec = trans_map.get((node, prev_arc.mode, arc.mode), {})
                t += max(0.0, safe_float(rec.get("time_h"), 0.0))

            event_key, bd = scenario_set.border_for_arc(arc, s)
            if event_key is not None and event_key not in incurred_border_events:
                incurred_border_events.add(event_key)
                if bd > 0.0:
                    t += bd
                    border_delay_h.setdefault(
                        node, np.zeros(size, dtype=float))[s] += bd

            entries = [] if arc.mode == "road" else \
                tt_dict.get((node, arc.to_node, arc.mode), [])
            if arc.mode != "road" and not entries:
                valid = False
                missing_timetable = True
                break
            departure = t if not entries else next_departure_time_programB(t, entries)
            wait_h = max(0.0, departure - t)
            if wait_h > 0.0:
                schedule_wait_h.setdefault(
                    node, np.zeros(size, dtype=float))[s] += wait_h

            travel_h = nominal_arc_travel_time(arc, tt_dict)
            travel_h *= scenario_set.travel(arc, s)
            t = departure + travel_h
            prev_arc = arc

        if valid:
            arrivals[s] = t

    result = PathScenarioResult(
        arrival_h=arrivals,
        schedule_wait_h=schedule_wait_h,
        border_delay_h=border_delay_h,
        missing_timetable=missing_timetable,
    )
    _PATH_SCENARIO_CACHE[cache_key] = result
    return result


def build_reliable_path_options(
    batches: List[Batch],
    path_lib: Dict[Tuple[str, str], List[Path]],
    tt_dict: Dict,
    trans_map: Dict,
    border_delay_map: Dict,
    scenario_set: ScenarioSet,
    mode: Optional[str] = None,
) -> Dict[Tuple[str, str, int], List[ReliablePathOption]]:
    """Pre-screen only timetable-valid, structurally usable paths.

    On-time probability and observed maximum lateness are retained solely as
    diagnostics and do not affect eligibility or feasibility-first seeding.
    ``mode`` is retained for call compatibility and has no screening effect.
    """

    options: Dict[Tuple[str, str, int], List[ReliablePathOption]] = {}
    counts: List[int] = []
    missing: List[int] = []

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        batch_options: List[ReliablePathOption] = []
        for path in path_lib.get((batch.origin, batch.destination), []):
            result = simulate_path_over_scenarios(
                path, batch, tt_dict, trans_map, border_delay_map,
                scenario_set)
            arrivals = result.arrival_h
            on_time_probability = float(np.mean(arrivals <= batch.LT))
            lateness = np.maximum(0.0, arrivals - batch.LT)
            finite_lateness = lateness[np.isfinite(lateness)]
            max_lateness = (float(np.max(finite_lateness))
                            if finite_lateness.size == lateness.size
                            else float("inf"))
            arc_flow: Dict = {}
            node_flow: Dict = {}
            duration, _, miss_tt = simulate_path_time_capacity(
                path, batch, batch.quantity, tt_dict, arc_flow,
                trans_map=trans_map,
                border_delay_map=border_delay_map,
                node_flow_map=node_flow,
                stochastic=False,
                record_capacity=True,
            )
            if miss_tt > 0 or math.isinf(duration):
                continue

            resources: Dict[Any, float] = {}
            for (arc_key, slot), flow in arc_flow.items():
                resources[("arc", arc_key, slot)] = float(flow)
            for (node, slot), flow in node_flow.items():
                resources[("node", node, slot)] = float(flow)

            batch_options.append(ReliablePathOption(
                path=path,
                on_time_probability=on_time_probability,
                max_lateness_h=max_lateness,
                resources=resources,
            ))

        batch_options.sort(key=lambda option: (
            -option.on_time_probability,
            option.max_lateness_h,
            option.path.base_travel_time_h,
            option.path.base_cost_per_teu,
        ))
        options[key] = batch_options
        counts.append(len(batch_options))
        if not batch_options:
            missing.append(batch.batch_id)

    if counts:
        print("[FEAS-INIT] Structurally feasible single-path options per batch: "
              f"min={min(counts)}  median={float(np.median(counts)):.1f}  "
              f"max={max(counts)}")
    if missing:
        print("[FEAS-INIT] WARNING: no structurally feasible path for "
              f"batch IDs {missing}. Capacity-aware seeding cannot guarantee "
              "a feasible individual for those batches.")
    return options


def _resource_available_capacity(resource: Any,
                                 arc_caps: Dict[Tuple[str, str, str], float]) -> float:
    if resource[0] == "arc":
        return max(0.0, float(arc_caps.get(resource[1], 1e9)))
    node = resource[1]
    capacity = float(BORDER_CAPACITY.get(node, 0.0))
    if capacity <= 0.0:
        return float("inf")
    return max(0.0, capacity - float(BACKGROUND_FLOW.get(node, 0.0)))


def _capacity_choice_score(
    choice: List[int],
    option_lists: List[List[ReliablePathOption]],
    arc_caps: Dict[Tuple[str, str, str], float],
) -> Tuple[float, Dict[Any, float], Dict[Any, List[int]]]:
    loads: Dict[Any, float] = {}
    contributors: Dict[Any, List[int]] = {}
    for batch_index, option_index in enumerate(choice):
        for resource, flow in option_lists[batch_index][option_index].resources.items():
            loads[resource] = loads.get(resource, 0.0) + flow
            contributors.setdefault(resource, []).append(batch_index)

    violations: Dict[Any, float] = {}
    for resource, load in loads.items():
        excess = load - _resource_available_capacity(resource, arc_caps)
        if excess > 1e-9:
            violations[resource] = excess
    return float(sum(violations.values())), violations, contributors


def _preferred_reliable_choice(
    individual: Optional[Individual],
    batches: List[Batch],
    option_lists: List[List[ReliablePathOption]],
) -> List[Optional[int]]:
    preferred: List[Optional[int]] = [None] * len(batches)
    if individual is None:
        return preferred
    for batch_index, batch in enumerate(batches):
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = sorted(
            individual.od_allocations.get(key, []),
            key=lambda alloc: alloc.share,
            reverse=True,
        )
        for alloc in allocs:
            match = next((i for i, option in enumerate(option_lists[batch_index])
                          if option.path == alloc.path), None)
            if match is not None:
                preferred[batch_index] = match
                break
    return preferred


def find_capacity_aware_choice(
    batches: List[Batch],
    reliable_options: Dict[Tuple[str, str, int], List[ReliablePathOption]],
    arc_caps: Dict[Tuple[str, str, str], float],
    initial_individual: Optional[Individual] = None,
    restarts: Optional[int] = None,
    iterations: Optional[int] = None,
) -> Tuple[Optional[List[int]], float]:
    """Random-restart min-conflicts search over structurally valid paths."""
    if restarts is None:
        restarts = FEASIBILITY_SEARCH_RESTARTS
    if iterations is None:
        iterations = FEASIBILITY_SEARCH_ITERATIONS
    option_lists = [reliable_options.get(
        (batch.origin, batch.destination, batch.batch_id), [])
        for batch in batches]
    if any(not batch_options for batch_options in option_lists):
        return None, float("inf")

    preferred = _preferred_reliable_choice(
        initial_individual, batches, option_lists)
    best_choice: Optional[List[int]] = None
    best_score = float("inf")

    for restart in range(max(1, int(restarts))):
        if restart == 0 and any(index is not None for index in preferred):
            choice = [
                index if index is not None else random.randrange(len(option_lists[i]))
                for i, index in enumerate(preferred)
            ]
        else:
            choice = [random.randrange(len(batch_options))
                      for batch_options in option_lists]

        for _ in range(max(1, int(iterations))):
            score, violations, contributors = _capacity_choice_score(
                choice, option_lists, arc_caps)
            if score < best_score - 1e-12:
                best_score = score
                best_choice = list(choice)
            if score <= 1e-9:
                return choice, 0.0

            conflict_batches = sorted({
                batch_index
                for resource in violations
                for batch_index in contributors.get(resource, [])
            })
            if not conflict_batches:
                break
            batch_index = random.choice(conflict_batches)
            current_option = choice[batch_index]
            candidates = []
            for option_index in range(len(option_lists[batch_index])):
                if option_index == current_option:
                    continue
                trial = list(choice)
                trial[batch_index] = option_index
                trial_score, _, _ = _capacity_choice_score(
                    trial, option_lists, arc_caps)
                candidates.append((trial_score, random.random(), option_index))
            if not candidates:
                break
            candidates.sort()

            # Prefer the best conflict-reducing move, while allowing a small
            # random walk to escape capacity plateaus.
            if candidates[0][0] <= score + 1e-12 or random.random() < 0.04:
                choice[batch_index] = candidates[0][2]
            else:
                break

    return best_choice, best_score


def capacity_aware_initial_individual(
    batches: List[Batch],
    reliable_options: Dict[Tuple[str, str, int], List[ReliablePathOption]],
    arc_caps: Dict[Tuple[str, str, str], float],
    initial_individual: Optional[Individual] = None,
) -> Tuple[Optional[Individual], float]:
    choice, excess = find_capacity_aware_choice(
        batches, reliable_options, arc_caps,
        initial_individual=initial_individual)
    if choice is None:
        return None, excess

    ind = Individual()
    for batch, option_index in zip(batches, choice):
        key = (batch.origin, batch.destination, batch.batch_id)
        option = reliable_options[key][option_index]
        ind.od_allocations[key] = [PathAllocation(
            path=option.path, share=1.0)]
    return ind, excess


def evaluate_individual(ind, *args, **kwargs):
    if ACTIVE_MUTATION_LOGGER is not None:
        return ACTIVE_MUTATION_LOGGER.evaluate(
            _evaluate_individual_impl, ind, *args, **kwargs)
    return _evaluate_individual_impl(ind, *args, **kwargs)


def _evaluate_individual_impl(
    ind, batches, arcs, tt_dict,
    waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
    node_hold_cost=None, node_proc_cost=None,
    carbon_tax_map=None, trans_map=None, border_delay_map=None,
    theta_rm=None, node_trans_cost=None,
):
    """Map a candidate solution to three deterministic empirical CCP values.

    Random variables are never sampled here.  The complete scenario set is
    generated once by ``configure_scenario_set`` and then treated as fixed data.
    """
    node_hold_cost   = node_hold_cost   or {}
    node_proc_cost   = node_proc_cost   or {}
    carbon_tax_map   = carbon_tax_map   or {}
    trans_map        = trans_map        or {}
    border_delay_map = border_delay_map or {}
    theta_rm         = theta_rm         or {}
    node_trans_cost  = node_trans_cost  or {}

    scenario_set = ACTIVE_SCENARIO_SET
    if scenario_set is None:
        scenario_set = configure_scenario_set(
            arcs, border_delay_map, MC_SCENARIOS, MC_BASE_SEED,
            stochastic=STOCHASTIC_EVAL)
    size = scenario_set.size
    cost_s = np.zeros(size, dtype=float)
    emission_s = np.zeros(size, dtype=float)
    makespan_s = np.zeros(size, dtype=float)

    arc_flow_map: Dict = {}
    node_flow_map: Dict = {}
    arc_caps = {(a.from_node, a.to_node, a.mode): a.capacity for a in arcs}

    miss_alloc = miss_tt = 0
    cap_excess = wait_teu_h_total = late_teu_h_total = 0.0
    trans_teu_h_total = trans_cost_total = carbon_cost_total = 0.0
    linehaul_carbon_cost_s = np.zeros(size, dtype=float)
    waiting_carbon_cost_s = np.zeros(size, dtype=float)
    min_on_time_prob = 1.0
    max_observed_late_h = mean_arrival_sum = 0.0
    batch_on_time_prob: Dict[int, float] = {}
    batch_max_lateness: Dict[int, float] = {}

    for batch in batches:
        key = (batch.origin, batch.destination, batch.batch_id)
        allocs = [a for a in ind.od_allocations.get(key, [])
                  if a.share > 1e-12]
        if not allocs:
            miss_alloc += 1
            continue

        batch_arrival_s = np.full(size, batch.ET, dtype=float)
        y_jmn_k: Dict[Tuple[str, str, str], float] = {}

        for alloc in allocs:
            flow = alloc.share * batch.quantity
            path = alloc.path

            # Capacity is a planning constraint and is checked once against the
            # nominal schedule.  Scenario delays affect performance objectives.
            travel_time, _, mtt = simulate_path_time_capacity(
                path, batch, flow, tt_dict, arc_flow_map,
                trans_map=trans_map, border_delay_map=border_delay_map,
                node_flow_map=node_flow_map, stochastic=False,
                record_capacity=True)
            if math.isinf(travel_time) or mtt > 0:
                miss_tt += max(1, mtt)
                batch_arrival_s[:] = float("inf")
                continue

            path_result = simulate_path_over_scenarios(
                path, batch, tt_dict, trans_map, border_delay_map,
                scenario_set)
            if path_result.missing_timetable:
                miss_tt += 1
            batch_arrival_s = np.maximum(
                batch_arrival_s, path_result.arrival_h)

            deterministic_cost = path.base_cost_per_teu * flow
            deterministic_emission = path.base_emission_per_teu * flow
            cost_s += deterministic_cost
            emission_s += deterministic_emission

            carbon_cost = 0.0
            for arc in path.arcs:
                region = getattr(arc, "from_region", "")
                if theta_rm.get((region, arc.mode), 1) == 0:
                    continue
                tax_rate = float(carbon_tax_map.get(region, 0.0))
                emission_tons = (
                    arc.emission_per_teu_km * arc.distance * flow / 1e6)
                carbon_cost += emission_tons * tax_rate
            cost_s += carbon_cost
            carbon_cost_total += carbon_cost
            linehaul_carbon_cost_s += carbon_cost

            for node, wait_values in path_result.schedule_wait_h.items():
                hold_rate = node_hold_cost.get(
                    node, waiting_cost_per_teu_h)
                cost_s += hold_rate * flow * wait_values
                wait_emission_g_s = (
                    wait_emis_g_per_teu_h * flow * wait_values)
                emission_s += wait_emission_g_s

                # The wait is for the path arc departing this node. Its
                # from-region and mode provide Scheme B's tax basis. Road
                # remains wait-free because the simulator records no road
                # schedule waiting.
                outgoing_arc = next(
                    (arc for arc in path.arcs if arc.from_node == node), None)
                if outgoing_arc is None:
                    raise RuntimeError(
                        f"schedule-wait node {node!r} has no outgoing path arc")
                wait_region = getattr(outgoing_arc, "from_region", "")
                wait_mode = outgoing_arc.mode
                wait_tax_rate = float(carbon_tax_map.get(wait_region, 0.0))
                wait_theta = theta_rm.get((wait_region, wait_mode), 1)
                wait_carbon_cost = (
                    wait_emission_g_s / 1e6 * wait_tax_rate * wait_theta)
                cost_s += wait_carbon_cost
                waiting_carbon_cost_s += wait_carbon_cost
                carbon_cost_total += float(np.mean(wait_carbon_cost))
                wait_teu_h_total += flow * float(np.mean(wait_values))

            for node, delay_values in path_result.border_delay_h.items():
                proc_rate = node_proc_cost.get(node, 0.0)
                cost_s += proc_rate * flow * delay_values

            path_lateness = np.maximum(0.0, path_result.arrival_h - batch.LT)
            cost_s += batch.penalty_per_teu_h * flow * path_lateness
            late_teu_h_total += flow * float(np.mean(path_lateness))

            for i in range(len(path.arcs) - 1):
                if path.arcs[i].mode != path.arcs[i + 1].mode:
                    node = path.arcs[i + 1].from_node
                    m_in = path.arcs[i].mode
                    m_out = path.arcs[i + 1].mode
                    y_jmn_k[(node, m_in, m_out)] = \
                        y_jmn_k.get((node, m_in, m_out), 0.0) + flow

        for (node, m_in, m_out), transfer_flow in y_jmn_k.items():
            rec = trans_map.get((node, m_in, m_out), {})
            unit_cost = safe_float(rec.get("cost_per_teu"), 0.0)
            transfer_h = safe_float(rec.get("time_h"), 0.0)
            transfer_cost = unit_cost * transfer_flow
            operation_cost = (
                node_trans_cost.get(node, 0.0) * transfer_flow * transfer_h)
            cost_s += transfer_cost + operation_cost
            trans_cost_total += transfer_cost + operation_cost
            trans_teu_h_total += transfer_flow * transfer_h

        on_time_probability = float(np.mean(batch_arrival_s <= batch.LT))
        batch_on_time_prob[int(batch.batch_id)] = on_time_probability
        min_on_time_prob = min(min_on_time_prob, on_time_probability)
        batch_lateness = np.maximum(0.0, batch_arrival_s - batch.LT)
        finite_lateness = batch_lateness[np.isfinite(batch_lateness)]
        if finite_lateness.size:
            batch_max_lateness[int(batch.batch_id)] = float(
                np.max(finite_lateness))
            max_observed_late_h = max(
                max_observed_late_h, float(np.max(finite_lateness)))
        else:
            batch_max_lateness[int(batch.batch_id)] = float("inf")
        finite_arrivals = batch_arrival_s[np.isfinite(batch_arrival_s)]
        if finite_arrivals.size:
            mean_arrival_sum += float(np.mean(finite_arrivals))
        makespan_s = np.maximum(makespan_s, batch_arrival_s)

    for (arc_key, slot), flow in arc_flow_map.items():
        capacity = arc_caps.get(arc_key, 1e9)
        if flow > capacity:
            cap_excess += flow - capacity

    border_cap_excess = 0.0
    max_border_util = 0.0
    border_node_flow: Dict[str, float] = {}
    border_node_util: Dict[str, float] = {}
    for (node, slot), flow in node_flow_map.items():
        border_node_flow[node] = border_node_flow.get(node, 0.0) + flow
        capacity = BORDER_CAPACITY.get(node, 0.0)
        if capacity <= 0.0:
            continue
        load = flow + BACKGROUND_FLOW.get(node, 0.0)
        utilisation = load / capacity
        border_node_util[node] = max(
            border_node_util.get(node, 0.0), utilisation)
        max_border_util = max(max_border_util, utilisation)
        if load > capacity:
            border_cap_excess += load - capacity

    # Diagnostic scenario means -- computed once here and reused both for
    # the EV objective branch below and for reporting, never recomputed.
    scenario_cost_mean     = float(np.mean(cost_s))
    scenario_emission_mean = float(np.mean(emission_s))
    scenario_time_mean     = float(np.mean(makespan_s))
    ind.cost_s = cost_s.copy()
    ind.emission_s = emission_s.copy()
    ind.makespan_s = makespan_s.copy()
    ind.linehaul_carbon_cost_s = linehaul_carbon_cost_s.copy()
    ind.waiting_carbon_cost_s = waiting_carbon_cost_s.copy()
    ind.carbon_cost_s = (
        linehaul_carbon_cost_s + waiting_carbon_cost_s).copy()
    ind.ev_objectives = (
        scenario_cost_mean, scenario_emission_mean, scenario_time_mean)

    f_cost, f_emission, f_time = aggregate_scenario_objectives(
        cost_s, emission_s, makespan_s,
        scenario_cost_mean, scenario_emission_mean, scenario_time_mean,
        RISK_METRIC, CONFIDENCE_COST, CONFIDENCE_EMISSION, CONFIDENCE_TIME,
    )

    penalty = (
        PEN_MISS_ALLOC * float(miss_alloc)
        + PEN_MISS_TT * float(miss_tt)
        + PEN_CAP_EXCESS_PER_TEU * float(cap_excess)
        + PEN_BORDER_CAP_EXCESS_PER_TEU * float(border_cap_excess)
    )

    ind.objectives = (f_cost, f_emission, f_time)
    ind.penalty = float(penalty)

    # Dimensionless violation measure for Deb-style constraint handling.
    # The legacy monetary penalty is retained for reporting, but no longer
    # determines selection between heterogeneous constraint classes.
    batch_count = max(1, len(batches))
    total_teu = max(1.0, float(sum(batch.quantity for batch in batches)))
    normalized_components = {
        "miss_alloc": float(miss_alloc) / batch_count,
        "miss_tt": float(miss_tt) / batch_count,
        "cap_excess": float(cap_excess) / total_teu,
        "border_cap_excess": float(border_cap_excess) / total_teu,
    }
    normalized_violation = float(sum(normalized_components.values()))
    ind.normalized_violation = normalized_violation
    hard_ok = (
        miss_alloc == 0 and miss_tt == 0
        and cap_excess <= 1e-9 and border_cap_excess <= 1e-9
    )
    ind.feasible_hard = bool(hard_ok)
    ind.feasible = bool(hard_ok)
    ind.vio_breakdown = {
        "miss_alloc": float(miss_alloc),
        "miss_tt": float(miss_tt),
        "cap_excess": float(cap_excess),
        "border_cap_excess": float(border_cap_excess),
        "max_border_util": float(max_border_util),
        "late_teu_h": float(late_teu_h_total),
        "max_observed_late_h": float(max_observed_late_h),
        "normalized_violation": normalized_violation,
        "min_on_time_prob": float(min_on_time_prob),
        "mean_arrival_h_sum": float(mean_arrival_sum),
        "wait_teu_h": float(wait_teu_h_total),
        "trans_teu_h": float(trans_teu_h_total),
        "trans_cost": float(trans_cost_total),
        "carbon_cost": float(carbon_cost_total),
        "scenario_cost_mean": scenario_cost_mean,
        "scenario_emission_mean": scenario_emission_mean,
        "scenario_time_mean": scenario_time_mean,
        "risk_metric": RISK_METRIC,
    }
    ind.batch_on_time_prob = batch_on_time_prob
    ind.batch_max_lateness_h = batch_max_lateness
    ind.border_flow = border_node_flow
    ind.border_util = border_node_util


# ════════════════════════════════════════════════════════
# GA operators
# ════════════════════════════════════════════════════════

def clone_gene(alloc):
    return PathAllocation(path=alloc.path, share=float(alloc.share))


def clear_inherited_evaluation_cache(*individuals):
    """A new offspring must receive its own final objective evaluation.

    ``deepcopy(parent)`` also copies logger-only memoisation attributes.  They
    are valid only for repeated calls on the same pipeline individual, not for
    a newly created offspring that happens to have an unchanged chromosome.
    """
    for individual in individuals:
        for attribute in ("_learning_eval_key", "_learning_eval_id"):
            if hasattr(individual, attribute):
                delattr(individual, attribute)


def crossover_structural(ind1, ind2, batches):
    """Uniform crossover at batch-gene level.

    Copying complete allocation blocks avoids the old slice concatenation,
    which could silently turn two single-path parents into a multi-path or
    empty allocation and sharply reduce batch on-time probability.
    """
    child1, child2 = Individual(), Individual()
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1  = ind1.od_allocations.get(key, [])
        g2  = ind2.od_allocations.get(key, [])
        if not g1 and not g2: continue
        source1, source2 = ((g1, g2) if random.random() < 0.5
                            else (g2, g1))
        if not source1: source1 = source2
        if not source2: source2 = source1
        child1.od_allocations[key] = [clone_gene(x) for x in source1]
        child2.od_allocations[key] = [clone_gene(x) for x in source2]
    return child1, child2


def path_from_arcs(new_arcs, origin, destination, path_id=-1,
                   node_region=None, tt_dict=None):
    if not new_arcs: return None
    nodes = [new_arcs[0].from_node] + [a.to_node for a in new_arcs]
    if nodes[0] != origin or nodes[-1] != destination: return None
    if len(set(nodes)) != len(nodes): return None
    if node_region is not None:
        if not china_border_monotone_ok(nodes, node_region): return None
        if not region_monotone_ok(nodes, node_region):       return None
    return Path(
        path_id=path_id, origin=origin, destination=destination,
        nodes=nodes, modes=[a.mode for a in new_arcs], arcs=new_arcs,
        base_cost_per_teu=sum(a.cost_per_teu_km * a.distance for a in new_arcs),
        base_emission_per_teu=sum(a.emission_per_teu_km * a.distance for a in new_arcs),
        base_travel_time_h=sum(
            nominal_arc_travel_time(a, tt_dict or {}) for a in new_arcs),
    )


def rebuild_path_from_nodes_modes(origin, destination, nodes, modes,
                                   tt_dict, arc_lookup, allow_road_fallback=True):
    if not nodes or len(nodes) < 2 or nodes[0] != origin or nodes[-1] != destination:
        return None
    if len(modes) != len(nodes) - 1 or len(set(nodes)) != len(nodes): return None
    new_arcs = []
    for i in range(len(modes)):
        u, v, m = nodes[i], nodes[i+1], modes[i]
        k = (u, v, m)
        if k not in arc_lookup: return None
        arc = arc_lookup[k]
        if arc.mode != "road" and not tt_dict.get((u, v, arc.mode), []):
            if allow_road_fallback and (u, v, "road") in arc_lookup:
                arc = arc_lookup[(u, v, "road")]
            else: return None
        new_arcs.append(arc)
    return path_from_arcs(new_arcs, origin, destination, tt_dict=tt_dict)


def find_common_internal_nodes(p1, p2):
    # Preserve chromosome order so a fixed RNG seed is reproducible across
    # processes regardless of Python's per-process hash seed.
    available = set(p2.nodes[1:-1])
    return list(dict.fromkeys(
        node for node in p1.nodes[1:-1] if node in available))


def perform_single_point_crossover_paths(pA, pB, join_node, tt_dict, arc_lookup):
    if join_node not in pA.nodes or join_node not in pB.nodes: return None
    ia, ib = pA.nodes.index(join_node), pB.nodes.index(join_node)
    return rebuild_path_from_nodes_modes(
        pA.origin, pA.destination,
        pA.nodes[:ia+1] + pB.nodes[ib+1:],
        pA.modes[:ia]   + pB.modes[ib:],
        tt_dict, arc_lookup)


def crossover_common_node(ind1, ind2, batches, tt_dict, arc_lookup):
    child1, child2 = Individual(), Individual()
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        g1  = ind1.od_allocations.get(key, [])
        g2  = ind2.od_allocations.get(key, [])
        if not g1 and not g2: continue
        if not g1:
            child1.od_allocations[key] = [clone_gene(x) for x in g2]
            child2.od_allocations[key] = [clone_gene(x) for x in g2]; continue
        if not g2:
            child1.od_allocations[key] = [clone_gene(x) for x in g1]
            child2.od_allocations[key] = [clone_gene(x) for x in g1]; continue
        c1_allocs = [clone_gene(x) for x in g1]
        c2_allocs = [clone_gene(x) for x in g2]
        idx1, idx2 = random.randrange(len(g1)), random.randrange(len(g2))
        p1, p2    = g1[idx1].path, g2[idx2].path
        common    = find_common_internal_nodes(p1, p2)
        if common:
            join = random.choice(common)
            np1  = perform_single_point_crossover_paths(p1, p2, join, tt_dict, arc_lookup)
            np2  = perform_single_point_crossover_paths(p2, p1, join, tt_dict, arc_lookup)
            if np1:
                c1_allocs[idx1] = PathAllocation(
                    path=np1, share=c1_allocs[idx1].share)
            if np2:
                c2_allocs[idx2] = PathAllocation(
                    path=np2, share=c2_allocs[idx2].share)
        child1.od_allocations[key] = merge_and_normalize(c1_allocs)
        child2.od_allocations[key] = merge_and_normalize(c2_allocs)
    return child1, child2


def crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup):
    if random.random() < CROSSOVER_SEGMENT_PROB:
        if (ACTIVE_MUTATION_LOGGER is not None
                and getattr(ACTIVE_MUTATION_LOGGER, "controls_crossover", False)):
            c1, c2 = ACTIVE_MUTATION_LOGGER.crossover_common_node(
                p1, p2, batches, tt_dict, arc_lookup)
        else:
            c1, c2 = crossover_common_node(p1, p2, batches, tt_dict, arc_lookup)
        if c1.od_allocations or c2.od_allocations: return c1, c2
    return crossover_structural(p1, p2, batches)


def random_initial_individual(batches, path_lib, max_paths=3):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        k      = random.randint(1, min(max_paths, len(paths)))
        chosen = random.sample(paths, k)
        raw    = [PathAllocation(path=p, share=random.random()) for p in chosen]
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = merge_and_normalize(raw)
    return ind


def greedy_initial_individual(batches, path_lib):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        best  = min(paths, key=lambda p: p.base_travel_time_h)
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = \
            [PathAllocation(path=best, share=1.0)]
    return ind

def greedy_initial_min_cost(batches, path_lib):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        best  = min(paths, key=lambda p: p.base_cost_per_teu)
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = \
            [PathAllocation(path=best, share=1.0)]
    return ind


def greedy_initial_min_emission(batches, path_lib):
    ind = Individual()
    for b in batches:
        paths = path_lib.get((b.origin, b.destination), [])
        if not paths: continue
        best  = min(paths, key=lambda p: p.base_emission_per_teu)
        ind.od_allocations[(b.origin, b.destination, b.batch_id)] = \
            [PathAllocation(path=best, share=1.0)]
    return ind

def mutate_add(ind, batch, path_lib):
    key    = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    pool   = path_lib.get((batch.origin, batch.destination), [])
    if not pool: return False
    cur   = {a.path for a in allocs}
    cands = [p for p in pool if p not in cur]
    if not cands:
        if allocs:
            allocs[random.randrange(len(allocs))] = PathAllocation(
                path=random.choice(pool), share=0.2)
            ind.od_allocations[key] = merge_and_normalize(allocs)
            return True
        return False
    allocs.append(PathAllocation(path=random.choice(cands), share=0.2))
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_del(ind, batch, target_index=None):
    key    = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if len(allocs) <= 1: return False
    allocs.pop(random.randrange(len(allocs)) if target_index is None else target_index)
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mod(ind, batch, target_index=None):
    key    = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs: return False
    chosen = random.choice(allocs) if target_index is None else allocs[target_index]
    chosen.share *= random.uniform(0.5, 1.5)
    ind.od_allocations[key] = merge_and_normalize(allocs)
    return True


def mutate_mode(ind, batch, tt_dict, arc_lookup, max_trials=20,
                target_index=None, target_arc=None):
    key    = (batch.origin, batch.destination, batch.batch_id)
    allocs = ind.od_allocations.get(key, [])
    if not allocs: return False
    idx       = random.randrange(len(allocs)) if target_index is None else target_index
    old_alloc = allocs[idx]
    p         = old_alloc.path
    if not p.arcs: return False
    arc_i     = random.randrange(len(p.arcs)) if target_arc is None else target_arc
    old_arc   = p.arcs[arc_i]
    u, v      = old_arc.from_node, old_arc.to_node
    for _ in range(max_trials):
        new_mode = random.choice([m for m in ["road","rail","water"] if m != old_arc.mode])
        k_arc    = (u, v, new_mode)
        if k_arc not in arc_lookup: continue
        if new_mode != "road" and not tt_dict.get((u, v, new_mode), []): continue
        new_arcs       = list(p.arcs)
        new_arcs[arc_i] = arc_lookup[k_arc]
        new_path        = path_from_arcs(
            new_arcs, p.origin, p.destination, tt_dict=tt_dict)
        if new_path is None: continue
        allocs_new      = deepcopy(allocs)
        allocs_new[idx] = PathAllocation(path=new_path, share=old_alloc.share)
        ind.od_allocations[key] = merge_and_normalize(allocs_new)
        return True
    return False


def mutate_replace_reliable(
    ind: Individual,
    batch: Batch,
    reliable_options: Optional[
        Dict[Tuple[str, str, int], List[ReliablePathOption]]],
    replacement_path_id: Optional[str] = None,
) -> bool:
    """Replace a batch's allocation by another structurally valid path."""
    if not reliable_options:
        return False
    key = (batch.origin, batch.destination, batch.batch_id)
    options = reliable_options.get(key, [])
    if not options:
        return False
    current_paths = {alloc.path for alloc in ind.od_allocations.get(key, [])}
    candidates = [option for option in options
                  if option.path not in current_paths]
    if not candidates:
        candidates = list(options)
    # Always consume Base's original path draw.  A learned joint action may
    # replace its result, but must not shift every later algorithm RNG draw.
    shadow_choice = random.choice(candidates)
    if replacement_path_id is None:
        chosen = shadow_choice
    else:
        chosen = next((option for option in candidates
                       if str(option.path.path_id) == str(replacement_path_id)), None)
        if chosen is None:
            return False
    ind.od_allocations[key] = [PathAllocation(
        path=chosen.path, share=1.0)]
    return True


_FIXED_OP_WEIGHTS = [W_ADD, W_DEL, W_MOD, W_MODE, W_REPLACE]
_FIXED_OP_TOTAL   = sum(_FIXED_OP_WEIGHTS)
_FIXED_OP_PROBS   = [w / _FIXED_OP_TOTAL for w in _FIXED_OP_WEIGHTS]


def sample_operator() -> str:
    r, cum = random.random(), 0.0
    for op, prob in zip(OPS, _FIXED_OP_PROBS):
        cum += prob
        if r <= cum: return op
    return OPS[-1]


def apply_mutation_op(ind, op, batch, path_lib, tt_dict, arc_lookup,
                      reliable_options=None, target=None):
    target_index = None if target is None else target.allocation_index
    target_arc = None if target is None else target.arc_index
    if op == "add":  return mutate_add(ind, batch, path_lib)
    if op == "del":  return mutate_del(ind, batch, target_index)
    if op == "mod":  return mutate_mod(ind, batch, target_index)
    if op == "mode": return mutate_mode(ind, batch, tt_dict, arc_lookup,
                                         target_index=target_index, target_arc=target_arc)
    if op == "replace":
        return mutate_replace_reliable(
            ind, batch, reliable_options,
            None if target is None else target.replacement_path_id)
    return False


def mutate_fixed(
    ind, batches, path_lib, tt_dict, arc_lookup,
    arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
    reliable_options=None,
    node_hold_cost=None, node_proc_cost=None,
    carbon_tax_map=None, trans_map=None, border_delay_map=None,
    theta_rm=None, node_trans_cost=None,
):
    if (ACTIVE_MUTATION_LOGGER is not None
            and getattr(ACTIVE_MUTATION_LOGGER, "controls_mutation", True)):
        return ACTIVE_MUTATION_LOGGER.mutate(
            ind, batches, path_lib, tt_dict, arc_lookup,
            arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
            reliable_options=reliable_options,
            node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
            carbon_tax_map=carbon_tax_map, trans_map=trans_map,
            border_delay_map=border_delay_map, theta_rm=theta_rm,
            node_trans_cost=node_trans_cost)
    batch = random.choice(batches)
    op    = sample_operator()
    ok    = apply_mutation_op(
        ind, op, batch, path_lib, tt_dict, arc_lookup,
        reliable_options=reliable_options)
    if not ok:
        return op, False
    repair_missing_allocations(ind, batches, path_lib)
    evaluate_individual(
        ind, batches, arcs, tt_dict,
        waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
        node_hold_cost=node_hold_cost, node_proc_cost=node_proc_cost,
        carbon_tax_map=carbon_tax_map, trans_map=trans_map,
        border_delay_map=border_delay_map,
        theta_rm=theta_rm, node_trans_cost=node_trans_cost,
    )
    return op, True


# ════════════════════════════════════════════════════════
# Dominance helper
# ════════════════════════════════════════════════════════

def dominates(a, b):
    if a.feasible and not b.feasible: return True
    if b.feasible and not a.feasible: return False
    if a.feasible and b.feasible:
        return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
                any(x <  y for x, y in zip(a.objectives, b.objectives)))
    # Deb-style constraint domination with a dimensionless aggregate.  The
    # previous monetary penalty made 1 TEU of capacity excess compete directly
    # with a probability shortfall, which distorted the route to feasibility.
    if a.normalized_violation < b.normalized_violation - 1e-12: return True
    if b.normalized_violation < a.normalized_violation - 1e-12: return False
    return (all(x <= y for x, y in zip(a.objectives, b.objectives)) and
            any(x <  y for x, y in zip(a.objectives, b.objectives)))


def constraint_sort_key(ind: Individual) -> Tuple[float, float, float]:
    return (
        0.0 if ind.feasible else 1.0,
        safe_float(ind.normalized_violation, float("inf")),
        safe_float(ind.penalty, float("inf")),
    )


def unique_individuals_by_objectives(front, tol=1e-3):
    uniq, seen = [], []
    for ind in front:
        obj = ind.objectives
        if not any(all(abs(obj[i]-o[i]) <= tol for i in range(NUM_OBJ)) for o in seen):
            seen.append(obj); uniq.append(ind)
    return uniq


def format_violation_breakdown(ind) -> str:
    keys = [
        "miss_alloc", "miss_tt", "cap_excess", "border_cap_excess",
        "late_teu_h",
        "normalized_violation", "min_on_time_prob",
    ]
    parts = []
    bd = getattr(ind, "vio_breakdown", {}) or {}
    for k in keys:
        if k in bd:
            parts.append(f"{k}={safe_float(bd.get(k), 0.0):.3g}")
    return "  ".join(parts) if parts else "no breakdown"


# ════════════════════════════════════════════════════════
# NSGA-II core components
# ════════════════════════════════════════════════════════

def fast_non_dominated_sort(pop: List[Individual]) -> List[List[int]]:
    """
    Deb et al. (2002) fast non-dominated sort.
    Returns fronts: list of index lists. fronts[0] = Pareto-front.
    Also writes ind.rank in-place (0-indexed).
    """
    N = len(pop)
    S         = [[] for _ in range(N)]
    n_dom     = [0]  * N
    rank      = [0]  * N
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q: continue
            if dominates(pop[p], pop[q]):
                S[p].append(q)
            elif dominates(pop[q], pop[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # drop trailing empty front

    for idx, r in enumerate(rank):
        pop[idx].rank = int(r)
    return fronts


def crowding_distance_assignment(pop: List[Individual],
                                  indices: List[int]) -> None:
    """Assign crowding distance to individuals in the given front (in place)."""
    if not indices:
        return
    L = len(indices)
    for idx in indices:
        pop[idx].crowding_distance = 0.0
    if L <= 2:
        for idx in indices:
            pop[idx].crowding_distance = float("inf")
        return

    for m in range(NUM_OBJ):
        sorted_idx = sorted(indices, key=lambda i: pop[i].objectives[m])
        f_min = pop[sorted_idx[0]].objectives[m]
        f_max = pop[sorted_idx[-1]].objectives[m]
        rng   = f_max - f_min
        pop[sorted_idx[0]].crowding_distance  = float("inf")
        pop[sorted_idx[-1]].crowding_distance = float("inf")
        if rng <= 1e-12:
            continue
        for k in range(1, L - 1):
            prev_o = pop[sorted_idx[k - 1]].objectives[m]
            next_o = pop[sorted_idx[k + 1]].objectives[m]
            pop[sorted_idx[k]].crowding_distance += (next_o - prev_o) / rng


def nsga2_environmental_selection(combined: List[Individual],
                                   pop_size: int) -> List[Individual]:
    """
    1. Non-dominated sort R = P ∪ Q
    2. Fill new population by fronts F1, F2, ... until next front overflows
    3. Splitting front: keep highest crowding distance
    """
    fronts = fast_non_dominated_sort(combined)
    new_pop: List[Individual] = []
    for front in fronts:
        if not front:
            continue
        crowding_distance_assignment(combined, front)
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(combined[i] for i in front)
            if len(new_pop) == pop_size:
                break
        else:
            remaining = pop_size - len(new_pop)
            front_sorted = sorted(
                front, key=lambda i: combined[i].crowding_distance, reverse=True)
            new_pop.extend(combined[i] for i in front_sorted[:remaining])
            break
    return new_pop[:pop_size]


def nsga2_binary_tournament(pop: List[Individual]) -> Individual:
    """Crowded-comparison tournament: lower rank wins; tie → larger crowding dist wins."""
    a, b = random.sample(pop, 2)
    if a.rank < b.rank: return a
    if b.rank < a.rank: return b
    return a if a.crowding_distance >= b.crowding_distance else b


# ════════════════════════════════════════════════════════
# Feasibility boost  (uses rank instead of SPEA2 fitness)
# ════════════════════════════════════════════════════════

def _select_boost_parents(pop, topk=FEASIBLE_BOOST_TOPK_PARENTS):
    feasible   = sorted([i for i in pop if i.feasible],
                        key=lambda x: (x.rank, -x.crowding_distance))
    if len(feasible) >= topk: return feasible[:topk]
    infeasible = sorted([i for i in pop if not i.feasible],
                        key=constraint_sort_key)
    return (feasible + infeasible)[:topk]


def feasibility_boost(
    population, batches, path_lib, tt_dict, arc_lookup,
    arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, eval_kwargs,
    reliable_options, arc_caps,
    boost_rounds=FEASIBLE_BOOST_ROUNDS,
    boost_mutation_rate=FEASIBLE_BOOST_MUTATION_RATE,
    topk_parents=FEASIBLE_BOOST_TOPK_PARENTS,
):
    parents  = _select_boost_parents(population, topk=topk_parents)
    new_inds = []
    capacity_seed_count = max(1, boost_rounds // 2)
    for _ in range(capacity_seed_count):
        initial = random.choice(parents) if parents else None
        ind, _ = capacity_aware_initial_individual(
            batches, reliable_options, arc_caps,
            initial_individual=initial)
        if ind is None:
            break
        evaluate_individual(
            ind, batches, arcs, tt_dict,
            waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
        new_inds.append(ind)

    remaining = max(0, boost_rounds - len(new_inds))
    if len(parents) < 2:
        greedy_funcs = [greedy_initial_min_cost,
                        greedy_initial_individual,
                        greedy_initial_min_emission]
        for r in range(remaining):
            ind = greedy_funcs[r % 3](batches, path_lib)
            repair_missing_allocations(ind, batches, path_lib)
            evaluate_individual(ind, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
            new_inds.append(ind)
    else:
        for _ in range(remaining):
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            clear_inherited_evaluation_cache(c1, c2)
            for child in (c1, c2):
                repair_missing_allocations(child, batches, path_lib)
                if random.random() < boost_mutation_rate:
                    mutate_fixed(
                        child, batches, path_lib, tt_dict, arc_lookup,
                        arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                        reliable_options=reliable_options, **eval_kwargs)
                # [V1 FIX] ALWAYS evaluate boost children. Previously, a child
                # that wasn't mutated kept the dataclass default penalty=0.0 /
                # feasible=False, polluting selection with fake "0-penalty"
                # individuals and stalling convergence whenever boost fired.
                evaluate_individual(
                    child, batches, arcs, tt_dict,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
                new_inds.append(child)
            if (ACTIVE_MUTATION_LOGGER is not None
                    and hasattr(ACTIVE_MUTATION_LOGGER, "settle_crossover")):
                ACTIVE_MUTATION_LOGGER.settle_crossover(c1, c2)
            if len(new_inds) >= boost_rounds: break

    pop_id_idx     = {id(ind): idx for idx, ind in enumerate(population)}
    infeas_sorted  = sorted([i for i in population if not i.feasible],
                            key=constraint_sort_key, reverse=True)
    new_sorted     = sorted(new_inds, key=constraint_sort_key)
    num_new_feas   = 0
    replaced_ids: set = set()

    for new_ind in new_sorted:
        for target in infeas_sorted:
            if id(target) in replaced_ids: continue
            if (new_ind.feasible or
                    new_ind.normalized_violation
                    < target.normalized_violation - 1e-12):
                idx = pop_id_idx.get(id(target))
                if idx is not None:
                    population[idx] = new_ind
                    replaced_ids.add(id(target))
                    if new_ind.feasible: num_new_feas += 1
                break
    return population, num_new_feas


# ════════════════════════════════════════════════════════
# HV calculator (Monte Carlo, 3-objective)
# ════════════════════════════════════════════════════════

class HypervolumeCalculator:
    def __init__(self, ref_point, num_samples=2000, seed=None):
        self.ref_point   = np.array(ref_point, dtype=float)
        self.num_samples = int(num_samples)
        rng = np.random.default_rng(seed)
        self.samples = rng.uniform(low=0.0, high=self.ref_point,
                                   size=(self.num_samples, NUM_OBJ))

    def calculate_points(self, points):
        if not points: return 0.0
        front = np.array(points, dtype=float)
        valid = front[np.all(front <= self.ref_point, axis=1)]
        if len(valid) == 0: return 0.0
        S   = self.samples[:, np.newaxis, :]
        O   = valid[np.newaxis, :, :]
        dom = np.any(np.all(O <= S, axis=2), axis=1)
        return float(np.sum(dom) / self.num_samples)


# ════════════════════════════════════════════════════════
# Metrics: P*, IGD+, Spacing
# ════════════════════════════════════════════════════════

def dominates_obj(a, b):
    return all(a[i] <= b[i] for i in range(NUM_OBJ)) and any(a[i] < b[i] for i in range(NUM_OBJ))


def nondominated_set(points):
    pts = unique_objective_tuples(points, tol=1e-9)
    return [p for i, p in enumerate(pts)
            if not any(dominates_obj(q, p) for j, q in enumerate(pts) if i != j)]


def normalize_points(points, mins, maxs):
    out = []
    for p in points:
        pp = []
        for i in range(NUM_OBJ):
            rng = maxs[i] - mins[i]
            pp.append(0.0 if rng <= 1e-12 else (p[i]-mins[i])/rng)
        out.append(tuple(pp))
    return out


def clip_points(points, ref):
    return [tuple(min(max(p[i], 0.0), ref[i]) for i in range(NUM_OBJ)) for p in points]


def igd_plus(P_star, A):
    if not P_star or not A: return float("inf")
    P, Q = np.array(P_star, dtype=float), np.array(A, dtype=float)
    return float(np.mean([
        float(np.min(np.sqrt(np.sum(np.maximum(Q - p, 0.0)**2, axis=1))))
        for p in P]))


def spacing_metric(A):
    if not A or len(A) < 2: return 0.0
    Q = np.array(A, dtype=float)
    n = Q.shape[0]
    dmin = []
    for i in range(n):
        diff = Q - Q[i]
        d    = np.sqrt(np.sum(diff**2, axis=1))
        d[i] = np.inf
        dmin.append(float(np.min(d)))
    dmin = np.array(dmin)
    return float(np.sqrt(np.sum((dmin - np.mean(dmin))**2) / max(1, n-1)))


def build_P_star_fast(run_front_hist, tail_gens=PSTAR_TAIL_GENS,
                      cap_per_gen=PSTAR_CAP_PER_GEN, max_total=PSTAR_MAX_TOTAL):
    pts = []
    for hist in run_front_hist:
        tail = hist[-tail_gens:] if tail_gens > 0 else hist
        for gen_front in tail:
            pts.extend(gen_front[:cap_per_gen])
            if len(pts) >= max_total: break
        if len(pts) >= max_total: break
    return nondominated_set(pts)


# ════════════════════════════════════════════════════════
# Export helpers
# ════════════════════════════════════════════════════════

def save_pareto_solutions(pareto, batches, filename="result.txt"):
    pareto = unique_individuals_by_objectives([i for i in pareto if i.feasible], tol=1e-3)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===== NSGA-II Tuned Baseline Pareto Solutions (3-Objective) =====\n\n")
        if not pareto:
            f.write("NO FEASIBLE SOLUTION FOUND.\n"); return
        for i, ind in enumerate(pareto):
            c, e, t = ind.objectives
            ev_c, ev_e, ev_t = ind.ev_objectives
            f.write(f"===== Pareto Sol {i} =====\n")
            f.write(f"CCP90_Cost={c:.6f}  CCP90_Emission={e:.6f}  "
                    f"CCP90_Time={t:.6f}\n")
            f.write(f"EV_Cost={ev_c:.6f}  EV_Emission={ev_e:.6f}  "
                    f"EV_Time={ev_t:.6f}\n")
            f.write(f"Penalty={ind.penalty:.6f}  Feasible={ind.feasible}\n")
            f.write(f"Breakdown={ind.vio_breakdown}\n\n")
            for b in batches:
                key    = (b.origin, b.destination, b.batch_id)
                allocs = ind.od_allocations.get(key, [])
                if not allocs: continue
                f.write(f"Batch {b.batch_id}: {b.origin} -> {b.destination}, Q={b.quantity}\n")
                for a in allocs: f.write(str(a) + "\n")
                f.write("\n")
            f.write("\n")
    print(f"[EXPORT] {len(pareto)} Pareto solutions → {filename}")


def _json_scalar(value, *, context: str = ""):
    """Type-aware JSON-scalar coercion for heterogeneous diagnostic dicts
    (currently: Individual.vio_breakdown, which legitimately mixes numeric
    diagnostics with a "risk_metric" string field -- see evaluate_individual).

    - bool is preserved as bool (checked BEFORE int/float: in Python
      `isinstance(True, int)` is True, so bool must be special-cased first
      or it would silently fall through the int/float branch instead).
    - NumPy integer/floating scalars are converted to native Python
      int/float, checked BEFORE the plain int/float branch: numpy.float64
      is (on common platforms) a *subclass* of Python's built-in float
      (`isinstance(np.float64(1.5), float)` is True), so checking
      plain (int, float) first would silently let a numpy scalar through
      un-coerced -- it would happen to round-trip correctly through
      json.dump/json.load, but the returned Python object itself would
      still be a numpy type, not the "native Python numeric value" this
      function is specified to produce. numpy.integer has no such
      subclass relationship with int, but numpy.floating is checked first
      for both, for one consistent, order-independent rule.
    - Python int/float (genuinely native, not numpy) are returned unchanged.
    - str is preserved as str.
    - None is preserved as None (JSON null).
    - Anything else is an explicit, loud failure -- NOT silently
      stringified or dropped, so an unexpected diagnostic value type is
      caught immediately rather than masked in the exported file.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return None
    raise TypeError(
        f"Unsupported diagnostic value type for JSON export"
        f"{f' ({context})' if context else ''}: "
        f"{type(value).__name__!r} = {value!r}. Add explicit handling in "
        f"_json_scalar() rather than coercing it implicitly.")


def export_pareto_points_json(pareto, batches, out_json="pareto_points.json"):
    out = []
    for ind in pareto:
        sol = {
            "objectives": {
                "CCP90_Cost":     float(ind.objectives[0]),
                "CCP90_Emission": float(ind.objectives[1]),
                "CCP90_Time":     float(ind.objectives[2]),
                "EV_Cost":        float(ind.ev_objectives[0]),
                "EV_Emission":    float(ind.ev_objectives[1]),
                "EV_Time":        float(ind.ev_objectives[2]),
                "penalty":       float(ind.penalty),
            },
            "feasible":      bool(ind.feasible),
            "vio_breakdown": {k: _json_scalar(v, context=f"vio_breakdown[{k!r}]")
                               for k, v in (ind.vio_breakdown or {}).items()},
            "batch_on_time_probability": {
                str(k): float(v) for k, v in ind.batch_on_time_prob.items()},
            "batch_max_lateness_h": {
                str(k): float(v) for k, v in ind.batch_max_lateness_h.items()},
            "border_flow":   {str(k): float(v) for k, v in getattr(ind, "border_flow", {}).items()},
            "border_util":   {str(k): float(v) for k, v in getattr(ind, "border_util", {}).items()},
            "allocations":   [],
        }
        for b in batches:
            key = (b.origin, b.destination, b.batch_id)
            blk = {
                "batch_id":    int(b.batch_id),
                "origin":      b.origin,
                "destination": b.destination,
                "quantity_teu": float(b.quantity),
                "ET":          float(b.ET),
                "LT":          float(b.LT),
                "paths":       [],
            }
            for a in ind.od_allocations.get(key, []):
                blk["paths"].append({
                    "share": float(a.share),
                    "nodes": list(a.path.nodes),
                    "modes": list(a.path.modes),
                    "base_cost_per_teu":     float(a.path.base_cost_per_teu),
                    "base_emission_per_teu": float(a.path.base_emission_per_teu),
                    "base_travel_time_h":    float(a.path.base_travel_time_h),
                })
            sol["allocations"].append(blk)
        out.append(sol)
    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] pareto_points.json → {out_json}  ({len(out)} solutions)")


def export_best_infeasible_json(population, batches, out_json="best_infeasible.json"):
    if not population:
        return
    best = min(population, key=constraint_sort_key)
    sol = {
        "objectives": {
            "CCP90_Cost":     float(best.objectives[0]),
            "CCP90_Emission": float(best.objectives[1]),
            "CCP90_Time":     float(best.objectives[2]),
            "EV_Cost":        float(best.ev_objectives[0]),
            "EV_Emission":    float(best.ev_objectives[1]),
            "EV_Time":        float(best.ev_objectives[2]),
            "penalty":       float(best.penalty),
        },
        "feasible": bool(best.feasible),
        "vio_breakdown": {k: _json_scalar(v, context=f"vio_breakdown[{k!r}]")
                           for k, v in (best.vio_breakdown or {}).items()},
        "batch_on_time_probability": {
            str(k): float(v) for k, v in best.batch_on_time_prob.items()},
        "batch_max_lateness_h": {
            str(k): float(v) for k, v in best.batch_max_lateness_h.items()},
        "border_flow": {str(k): float(v) for k, v in getattr(best, "border_flow", {}).items()},
        "border_util": {str(k): float(v) for k, v in getattr(best, "border_util", {}).items()},
        "allocations": [],
    }
    for b in batches:
        key = (b.origin, b.destination, b.batch_id)
        blk = {
            "batch_id": int(b.batch_id),
            "origin": b.origin,
            "destination": b.destination,
            "quantity_teu": float(b.quantity),
            "ET": float(b.ET),
            "LT": float(b.LT),
            "paths": [],
        }
        for a in best.od_allocations.get(key, []):
            blk["paths"].append({
                "share": float(a.share),
                "nodes": list(a.path.nodes),
                "modes": list(a.path.modes),
                "base_cost_per_teu": float(a.path.base_cost_per_teu),
                "base_emission_per_teu": float(a.path.base_emission_per_teu),
                "base_travel_time_h": float(a.path.base_travel_time_h),
            })
        sol["allocations"].append(blk)
    with open(out_json, "w", encoding="utf-8") as f:
        _json.dump(sol, f, ensure_ascii=False, indent=2)
    print(f"[EXPORT] best_infeasible.json → {out_json}")


# ════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════

def _band(ax, gen, mean, std, color, alpha=0.18):
    ax.fill_between(gen, mean - std, mean + std, alpha=alpha, color=color)


def plot_pareto_3d(pts, save, title="NSGA-II Pareto Front 3D"):
    if not pts: return
    A = _finite_points_array(pts)
    if A.shape[0] == 0: return
    fig = plt.figure(figsize=(7, 6), dpi=200)
    ax  = fig.add_subplot(111, projection="3d")
    sc  = ax.scatter(A[:, 0], A[:, 1], A[:, 2], c=A[:, 0], cmap="viridis", s=40, alpha=0.9)
    plt.colorbar(sc, ax=ax, pad=0.1, fraction=0.04, label="Cost ($)")
    ax.set_xlabel("Cost ($)"); ax.set_ylabel("Emission (gCO₂)"); ax.set_zlabel("Time (h)")
    ax.set_title(title); ax.grid(True, ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


def plot_pareto_2d(pts, save):
    if not pts: return
    A = _finite_points_array(pts)
    if A.shape[0] == 0: return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=180)
    pairs = [(0,1,"Cost ($)","Emission (gCO₂)"),
             (0,2,"Cost ($)","Time (h)"),
             (1,2,"Emission (gCO₂)","Time (h)")]
    for ax, (xi, yi, xl, yl) in zip(axes, pairs):
        ax.scatter(A[:, xi], A[:, yi], s=30, alpha=0.85, c=A[:, 0], cmap="viridis")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{xl} vs {yl}")
        ax.grid(True, ls=":", alpha=0.45)
    fig.suptitle("NSGA-II Final Pareto Front — 2D Projections", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(save); plt.close()
    print(f"[PLOT] {save}")


# ════════════════════════════════════════════════════════
# Core NSGA-II runner
# ════════════════════════════════════════════════════════

def _active_logger_budget_reserve(stage, units=1):
    """Let an attached controller declare its real-evaluation requirement.

    Older loggers have no declaration and retain the historical conservative
    four-evaluations-per-unit guard.
    """
    if ACTIVE_MUTATION_LOGGER is None:
        return 0
    reserve = getattr(ACTIVE_MUTATION_LOGGER, "budget_reserve", None)
    return int(reserve(stage, units)) if reserve is not None else 4 * int(units)

def run_nsga2(
    node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
    arcs, timetables, batches,
    waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
    carbon_tax_map, emission_factor_map, mode_speeds_map,
    trans_map, border_delay_map, theta_rm, path_lib, reliable_options,
    pop_size=250, generations=200,
    archive_size=None,
):
    if archive_size is None:
        archive_size = pop_size

    tt_dict    = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)
    arc_caps   = {(arc.from_node, arc.to_node, arc.mode): arc.capacity
                  for arc in arcs}

    eval_kwargs = dict(
        node_hold_cost=node_hold_cost,
        node_proc_cost=node_proc_cost,
        carbon_tax_map=carbon_tax_map,
        trans_map=trans_map,
        border_delay_map=border_delay_map,
        theta_rm=theta_rm,
        node_trans_cost=node_trans_cost,
    )

    # ── Initialise population: capacity-aware feasible seeds + diversity ─
    requested_seeds = int(round(pop_size * FEASIBILITY_SEED_FRACTION))
    if FEASIBILITY_SEED_FRACTION > 0.0 and requested_seeds == 0:
        requested_seeds = 1
    n_feasibility_seeds = min(pop_size, max(0, requested_seeds))
    non_seed_count = max(0, pop_size - n_feasibility_seeds)
    n_each = max(1, non_seed_count // 6) if non_seed_count else 0
    population: List[Individual] = []
    seed_zero_excess = 0
    for i in range(pop_size):
        if i < n_feasibility_seeds:
            ind, capacity_excess = capacity_aware_initial_individual(
                batches, reliable_options, arc_caps)
            if ind is None:
                ind = random_initial_individual(batches, path_lib, max_paths=1)
            elif capacity_excess <= 1e-9:
                seed_zero_excess += 1
        elif i < n_feasibility_seeds + n_each:
            ind = greedy_initial_min_cost(batches, path_lib)
        elif i < n_feasibility_seeds + 2 * n_each:
            ind = greedy_initial_individual(batches, path_lib)        # min-time
        elif i < n_feasibility_seeds + 3 * n_each:
            ind = greedy_initial_min_emission(batches, path_lib)
        else:
            if (ACTIVE_MUTATION_LOGGER is not None
                    and getattr(
                        ACTIVE_MUTATION_LOGGER, "controls_initialization", False)):
                ind = ACTIVE_MUTATION_LOGGER.initial_individual(
                    batches, path_lib, max_paths=3)
            else:
                ind = random_initial_individual(batches, path_lib)
        repair_missing_allocations(ind, batches, path_lib)
        evaluate_individual(ind, batches, arcs, tt_dict,
                        waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
        if (ACTIVE_MUTATION_LOGGER is not None
                and hasattr(ACTIVE_MUTATION_LOGGER, "settle_initialization")):
            ACTIVE_MUTATION_LOGGER.settle_initialization(ind)
        population.append(ind)

    initial_feasible = sum(1 for ind in population if ind.feasible)
    print(f"[FEAS-INIT] Capacity-aware seeds={n_feasibility_seeds}  "
          f"zero-excess={seed_zero_excess}  "
          f"fully-feasible-after-evaluation={initial_feasible}")

    # Initial sort + crowding so first tournament has valid keys
    fronts0 = fast_non_dominated_sort(population)
    for front in fronts0:
        crowding_distance_assignment(population, front)
    if (ACTIVE_MUTATION_LOGGER is not None
            and hasattr(ACTIVE_MUTATION_LOGGER, "log_initial_population")):
        ACTIVE_MUTATION_LOGGER.log_initial_population(population)

    front_hist_objs:            List[List[Tuple]] = []
    feasible_ratio_hist:        List[float]       = []
    feasible_ratio_strict_hist: List[float]       = []
    vio_mean_hist = {k: [] for k in [
        "miss_alloc", "miss_tt", "cap_excess", "border_cap_excess",
        "late_teu_h", "wait_teu_h",
        "normalized_violation",
    ]}
    boost_trigger_hist:  List[int] = []
    boost_new_feas_hist: List[int] = []
    pareto_size_hist:    List[int] = []
    feasible_count_hist: List[int] = []

    _run_start = time.perf_counter()
    _prev_best = [float("inf")] * NUM_OBJ

    for gen in range(generations):
        if ACTIVE_MUTATION_LOGGER is not None:
            ACTIVE_MUTATION_LOGGER.generation = gen
            ACTIVE_MUTATION_LOGGER.phase = "offspring"
            if not ACTIVE_MUTATION_LOGGER.has_budget(
                    _active_logger_budget_reserve("offspring_pair")):
                break

        # ── Step 1: Binary tournament → mating pool ──────
        mating_pool = [nsga2_binary_tournament(population) for _ in range(pop_size)]

        # ── Step 2: Crossover + Mutation → offspring Q ───
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            if (ACTIVE_MUTATION_LOGGER is not None
                    and not ACTIVE_MUTATION_LOGGER.has_budget(
                        _active_logger_budget_reserve("offspring_pair"))):
                break
            p1, p2 = random.sample(mating_pool, 2)
            if random.random() < CROSSOVER_RATE:
                c1, c2 = crossover_hybrid(p1, p2, batches, tt_dict, arc_lookup)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            clear_inherited_evaluation_cache(c1, c2)

            for child in (c1, c2):
                repair_missing_allocations(child, batches, path_lib)

            if (ACTIVE_MUTATION_LOGGER is not None
                    and hasattr(
                        ACTIVE_MUTATION_LOGGER,
                        "evaluate_crossover_counterfactual")):
                ACTIVE_MUTATION_LOGGER.evaluate_crossover_counterfactual(
                    _evaluate_individual_impl, c1, c2, batches, arcs, tt_dict,
                    waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                    **eval_kwargs)

            if random.random() < MUTATION_RATE:
                mutate_fixed(c1, batches, path_lib, tt_dict, arc_lookup,
                             arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                             reliable_options=reliable_options, **eval_kwargs)
            if random.random() < MUTATION_RATE:
                mutate_fixed(c2, batches, path_lib, tt_dict, arc_lookup,
                             arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
                             reliable_options=reliable_options, **eval_kwargs)

            repair_missing_allocations(c1, batches, path_lib)
            repair_missing_allocations(c2, batches, path_lib)
            evaluate_individual(c1, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
            evaluate_individual(c2, batches, arcs, tt_dict,
                                waiting_cost_per_teu_h, wait_emis_g_per_teu_h, **eval_kwargs)
            if (ACTIVE_MUTATION_LOGGER is not None
                    and hasattr(ACTIVE_MUTATION_LOGGER, "settle_crossover")):
                ACTIVE_MUTATION_LOGGER.settle_crossover(c1, c2)
            offspring.extend([c1, c2])
        offspring = offspring[:pop_size]

        # ── Step 3: Environmental selection (P ∪ Q → new P) ──
        combined   = population + offspring
        population = nsga2_environmental_selection(combined, pop_size)
        if ACTIVE_MUTATION_LOGGER is not None:
            ACTIVE_MUTATION_LOGGER.flush_events(population)

        # ── Record metrics from population ───────────────
        feas_pop     = [ind for ind in population if ind.feasible]
        non_dom_feas = [ind for ind in population if ind.rank == 0 and ind.feasible]
        display      = unique_individuals_by_objectives(non_dom_feas, tol=1e-3)

        front_hist_objs.append([ind.objectives for ind in display])
        feas_total = len(feas_pop)
        feasible_ratio_hist.append(feas_total / max(1, pop_size))
        feasible_ratio_strict_hist.append(
            sum(1 for i in population if i.feasible_hard) / max(1, pop_size))
        pareto_size_hist.append(len(display))
        feasible_count_hist.append(feas_total)

        for k in vio_mean_hist:
            vals = [ind.vio_breakdown.get(k, 0.0) for ind in population]
            vio_mean_hist[k].append(float(np.mean(vals)))

        elapsed  = time.perf_counter() - _run_start
        if feas_pop:
            cur    = [min(i.objectives[j] for i in feas_pop) for j in range(NUM_OBJ)]
            d      = ["↓" if cur[j] < _prev_best[j] - 1e-3 else "→" for j in range(NUM_OBJ)]
            _prev_best = cur
            obj_str = (f"Cost={cur[0]:.3e}{d[0]} "
                       f"Emis={cur[1]:.3e}{d[1]} "
                       f"Time={cur[2]:.1f}h{d[2]}")
        else:
            obj_str = "No feasible solutions yet"

        best_pen_ind = min(population, key=constraint_sort_key) if population else None
        best_pen = best_pen_ind.penalty if best_pen_ind is not None else float("inf")
        sep      = "=" * 72
        print(f"\n{sep}")
        print(f"  [NSGA-II] Gen {gen:03d}/{generations-1}  |  {elapsed:.1f}s elapsed")
        print(f"  Pop feasible: {feas_total}/{pop_size} "
              f"({feasible_ratio_hist[-1]:.1%})"
              f"  |  NonDom={len(display)}"
              f"  |  BestPenalty={best_pen:.2e}"
              f"  |  BestNormVio={best_pen_ind.normalized_violation:.3g}")
        print(f"  Best feasible: {obj_str}")
        if best_pen_ind is not None and not best_pen_ind.feasible:
            print(f"  Best violation: {format_violation_breakdown(best_pen_ind)}")
        print(sep)

        # ── Step 4: Feasibility boost ────────────────────
        boost_triggered = boost_new_feas = 0
        if (feas_total < MIN_FEASIBLE_SOLUTIONS
                and (ACTIVE_MUTATION_LOGGER is None
                     or ACTIVE_MUTATION_LOGGER.has_budget(
                         _active_logger_budget_reserve(
                             "feasibility_boost", FEASIBLE_BOOST_ROUNDS)))):
            if ACTIVE_MUTATION_LOGGER is not None:
                ACTIVE_MUTATION_LOGGER.phase = "boost"
            boost_triggered = 1
            population, boost_new_feas = feasibility_boost(
                population, batches, path_lib, tt_dict, arc_lookup,
                arcs, waiting_cost_per_teu_h, wait_emis_g_per_teu_h, eval_kwargs,
                reliable_options, arc_caps)
            # Re-rank after boost so next tournament uses fresh keys
            fronts_after = fast_non_dominated_sort(population)
            for front in fronts_after:
                crowding_distance_assignment(population, front)
            after = sum(1 for i in population if i.feasible)
            print(f"  ⚡ [BOOST] {feas_total} < {MIN_FEASIBLE_SOLUTIONS} "
                  f"→ after boost: {after} (+{boost_new_feas})")

        boost_trigger_hist.append(boost_triggered)
        boost_new_feas_hist.append(boost_new_feas)
        if ACTIVE_MUTATION_LOGGER is not None:
            ACTIVE_MUTATION_LOGGER.flush_events(population)
            ACTIVE_MUTATION_LOGGER.log_generation(population, boost_triggered)

    # ── Final Pareto: rank-0 feasible from population ────────
    fronts_final = fast_non_dominated_sort(population)
    for front in fronts_final:
        crowding_distance_assignment(population, front)

    pareto = unique_individuals_by_objectives(
        [ind for ind in population if ind.rank == 0 and ind.feasible], tol=1e-3)
    if not pareto:
        feas_remaining = [ind for ind in population if ind.feasible]
        if feas_remaining:
            pareto = unique_individuals_by_objectives(
                sorted(feas_remaining, key=lambda x: (x.rank, -x.crowding_distance)),
                tol=1e-3)

    total_t = time.perf_counter() - _run_start
    print(f"\n{'='*72}")
    completed_generations = (ACTIVE_MUTATION_LOGGER.generations_completed
                             if ACTIVE_MUTATION_LOGGER is not None else generations)
    print(f"  [NSGA-II] Run complete: {completed_generations} gens, {total_t:.1f}s, Pareto={len(pareto)}")
    print(f"  ⚡ Boost: {sum(boost_trigger_hist)} gens triggered, "
          f"{sum(boost_new_feas_hist)} new feasible")
    if not pareto and population:
        best_final = min(population, key=constraint_sort_key)
        print(f"  Best infeasible penalty: {best_final.penalty:.2e}")
        print(f"  Best infeasible violation: {format_violation_breakdown(best_final)}")
    print(f"{'='*72}")

    return (
        population, pareto,
        front_hist_objs,
        feasible_ratio_hist, feasible_ratio_strict_hist,
        vio_mean_hist,
        boost_trigger_hist, boost_new_feas_hist,
        pareto_size_hist, feasible_count_hist,
        total_t,
    )


# ════════════════════════════════════════════════════════
# Main  ── NSGA-II Tuned Baseline 30 runs
# ════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the production CLI parser.

    Pulled out of the `if __name__ == "__main__":` block so tests can
    exercise the REAL production parser (e.g. `--risk-metric`,
    `--road-speed-kmh`) directly, rather than a hand-reconstructed
    approximation of it. Pure construction only -- no argument values are
    read or applied here; that still happens in `__main__` via
    `parser.parse_args()`. No behaviour change: every `add_argument` call
    below is unchanged from before this refactor.
    """
    parser = argparse.ArgumentParser(description=(
        "Run NSGA-II with stochastic travel time, border delay, and "
        "quantile-based objective optimisation."))
    parser.add_argument("--data", default="data_expanded.xlsx", help="Input workbook path.")
    parser.add_argument("--pop", type=int, default=250, help="Population size.")
    parser.add_argument("--gens", type=int, default=200, help="Number of generations.")
    parser.add_argument("--runs", type=int, default=30, help="Number of random-seed runs.")
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed.")
    parser.add_argument("--out", default="out_uncertainty", help="Output directory.")
    parser.add_argument("--batches-csv", default="",
                        help="Optional shared demand CSV. When set, overrides workbook Batches.")
    parser.add_argument("--expected-batches", type=int, default=0,
                        help="Optional exact batch count check; 0 disables the check.")
    parser.add_argument("--no-stochastic", action="store_true",
                        help="Disable Monte Carlo scenario evaluation.")
    parser.add_argument("--mc-scenarios", type=int, default=MC_SCENARIOS,
                        help="Number of frozen training scenarios shared by all individuals.")
    parser.add_argument("--mc-seed", type=int, default=MC_BASE_SEED,
                        help="Base seed for deterministic common random numbers.")
    parser.add_argument("--cost-confidence", type=float, default=CONFIDENCE_COST,
                        help="CCP confidence level for total cost.")
    parser.add_argument("--emission-confidence", type=float, default=CONFIDENCE_EMISSION,
                        help="CCP confidence level for total emissions.")
    parser.add_argument("--time-confidence", type=float, default=CONFIDENCE_TIME,
                        help="CCP confidence level for makespan.")
    parser.add_argument("--road-cv", type=float, default=MODE_TIME_CV["road"],
                        help="Road travel-time lognormal multiplier CV.")
    parser.add_argument("--rail-cv", type=float, default=MODE_TIME_CV["rail"],
                        help="Rail travel-time lognormal multiplier CV.")
    parser.add_argument("--water-cv", type=float, default=MODE_TIME_CV["water"],
                        help="Water travel-time lognormal multiplier CV.")
    parser.add_argument("--road-max-factor", type=float,
                        default=MODE_TIME_MAX_FACTOR["road"])
    parser.add_argument("--rail-max-factor", type=float,
                        default=MODE_TIME_MAX_FACTOR["rail"])
    parser.add_argument("--water-max-factor", type=float,
                        default=MODE_TIME_MAX_FACTOR["water"])
    parser.add_argument("--border-cv", type=float, default=None,
                        help="Compatibility option: use one border-delay CV for every mode.")
    parser.add_argument("--border-road-cv", type=float,
                        default=BORDER_DELAY_CV["road"])
    parser.add_argument("--border-rail-cv", type=float,
                        default=BORDER_DELAY_CV["rail"])
    parser.add_argument("--border-water-cv", type=float,
                        default=BORDER_DELAY_CV["water"])
    parser.add_argument("--border-max-factor", type=float, default=BORDER_DELAY_MAX_FACTOR,
                        help="Upper bound for sampled border delay as factor of the BCP-pair mean.")
    parser.add_argument(
        "--border-event-data", default=DEFAULT_BORDER_EVENT_DATA_FILE,
        help="CSV with directional b=(exit, entry, mode) border-event means.")
    parser.add_argument(
        "--road-speed-kmh", type=positive_speed_kmh_arg, default=None,
        help="Explicit road speed override (km/h); must be > 0 -- zero or "
             "negative values are rejected with a CLI error, not silently "
             "ignored. Takes precedence over the workbook's Mode_Speeds "
             "sheet; omit to let the workbook decide (fallback if neither "
             f"is available: {DEFAULT_MODE_SPEED_KMH['road']:g} km/h).")
    parser.add_argument(
        "--rail-speed-kmh", type=positive_speed_kmh_arg, default=None,
        help="Explicit rail speed override (km/h); must be > 0 -- zero or "
             "negative values are rejected with a CLI error, not silently "
             "ignored. Takes precedence over the workbook's Mode_Speeds "
             "sheet; omit to let the workbook decide (fallback if neither "
             f"is available: {DEFAULT_MODE_SPEED_KMH['rail']:g} km/h).")
    parser.add_argument(
        "--water-speed-kmh", type=positive_speed_kmh_arg, default=None,
        help="Explicit water speed override (km/h); must be > 0 -- zero or "
             "negative values are rejected with a CLI error, not silently "
             "ignored. Takes precedence over the workbook's Mode_Speeds "
             "sheet; omit to let the workbook decide (fallback if neither "
             f"is available: {DEFAULT_MODE_SPEED_KMH['water']:g} km/h).")
    parser.add_argument(
        "--risk-metric", choices=["deterministic", "ev", "ccp"], default="ccp",
        help="Evaluation mode reducing per-scenario objective arrays to "
             "fitness values: deterministic (single nominal scenario, "
             "S=1), ev (direct expected inputs), or ccp (separate empirical "
             "objective quantiles; no punctuality chance constraint).")
    parser.add_argument(
        "--feasible-seed-fraction", type=float,
        default=FEASIBILITY_SEED_FRACTION,
        help="Initial-population fraction built by capacity-aware structural coordination.")
    parser.add_argument(
        "--feasibility-restarts", type=int,
        default=FEASIBILITY_SEARCH_RESTARTS,
        help="Random restarts for capacity-aware feasible construction.")
    parser.add_argument(
        "--feasibility-iterations", type=int,
        default=FEASIBILITY_SEARCH_ITERATIONS,
        help="Min-conflicts iterations per feasibility restart.")
    parser.add_argument(
        "--late-penalty-usd-per-teu-h", type=float,
        default=DEFAULT_LATE_PENALTY_USD_PER_TEU_H,
        help="Baseline late-delivery penalty in USD/TEU/hour.")
    parser.add_argument(
        "--late-penalty-usd-per-teu-day", type=float, default=None,
        help="Legacy daily-rate override; when set it is divided by 24 and "
             "takes precedence over --late-penalty-usd-per-teu-h.")
    parser.add_argument(
        "--use-input-late-penalties", action="store_true",
        help="Use per-batch hourly penalties from the workbook/CSV instead of the sourced baseline.")
    parser.add_argument(
        "--payload-tonnes-per-teu", type=float, default=PAYLOAD_TONNES_PER_TEU,
        help="Payload assumption used to convert gCO2/t-km to gCO2/TEU-km.")
    parser.add_argument(
        "--wait-emission-g-per-teu-h", type=float, default=None,
        help="Optional waiting/idling emission override; zero leaves emissions deterministic.")
    return parser


if __name__ == "__main__":

    parser = build_arg_parser()
    args = parser.parse_args()

    DATA_FILE    = args.data
    POP_SIZE     = args.pop
    GENERATIONS  = args.gens
    RUNS         = args.runs
    BASE_SEED    = args.seed
    OUTPUT_DIR   = args.out
    RISK_METRIC = args.risk_metric
    STOCHASTIC_EVAL, MC_SCENARIOS = resolve_stochastic_eval_and_scenarios(
        RISK_METRIC, args.no_stochastic, args.mc_scenarios)
    MC_BASE_SEED = int(args.mc_seed)
    CONFIDENCE_COST = min(1.0, max(0.0, float(args.cost_confidence)))
    CONFIDENCE_EMISSION = min(1.0, max(0.0, float(args.emission_confidence)))
    CONFIDENCE_TIME = min(1.0, max(0.0, float(args.time_confidence)))
    BORDER_DELAY_MAX_FACTOR = max(1.0, float(args.border_max_factor))
    MODE_TIME_CV = {
        "road": max(0.0, float(args.road_cv)),
        "rail": max(0.0, float(args.rail_cv)),
        "water": max(0.0, float(args.water_cv)),
    }
    MODE_TIME_MAX_FACTOR = {
        "road": max(1.0, float(args.road_max_factor)),
        "rail": max(1.0, float(args.rail_max_factor)),
        "water": max(1.0, float(args.water_max_factor)),
    }
    # None unless the user explicitly passed --{mode}-speed-kmh; resolved
    # against the workbook and the hardcoded fallback inside
    # load_network_from_extended() via resolve_effective_speed_map().
    CLI_MODE_SPEED_OVERRIDE_KMH = {
        "road": args.road_speed_kmh,
        "rail": args.rail_speed_kmh,
        "water": args.water_speed_kmh,
    }
    BORDER_EVENT_DEFINITIONS = load_border_event_definitions(
        args.border_event_data)
    if args.border_cv is not None:
        common_border_cv = max(0.0, float(args.border_cv))
        BORDER_DELAY_CV = {m: common_border_cv for m in ("road", "rail", "water")}
    else:
        BORDER_DELAY_CV = {
            "road": max(0.0, float(args.border_road_cv)),
            "rail": max(0.0, float(args.border_rail_cv)),
            "water": max(0.0, float(args.border_water_cv)),
        }
    FEASIBILITY_SEED_FRACTION = min(
        1.0, max(0.0, float(args.feasible_seed_fraction)))
    FEASIBILITY_SEARCH_RESTARTS = max(1, int(args.feasibility_restarts))
    FEASIBILITY_SEARCH_ITERATIONS = max(
        1, int(args.feasibility_iterations))
    PAYLOAD_TONNES_PER_TEU = max(0.0, float(args.payload_tonnes_per_teu))
    if args.late_penalty_usd_per_teu_day is not None:
        sourced_late_penalty_h = max(
            0.0, float(args.late_penalty_usd_per_teu_day)) / 24.0
        late_penalty_input_basis = "legacy_daily_rate"
    else:
        sourced_late_penalty_h = max(
            0.0, float(args.late_penalty_usd_per_teu_h))
        late_penalty_input_basis = "hourly_rate"

    FSPath(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  NSGA-II  STOCHASTIC EXTENSION  (3-objective)")
    print("  ▶ Frozen scenarios + empirical CCP quantiles")
    print("=" * 72)
    print(f"  GA      : pop={POP_SIZE}  gen={GENERATIONS}  runs={RUNS}")
    print(f"            pc={CROSSOVER_RATE}  pm={MUTATION_RATE}")
    print(f"  PathLib : DFS={DFS_MAX_PATHS_PER_OD}  TopK={PATHS_TOPK_PER_CRITERION}  "
          f"LibCap={PATH_LIB_CAP_TOTAL}")
    print(f"  HV      : MC samples={HV_SAMPLES}  ref={HV_REF_NORM}")
    print(f"  Data    : {DATA_FILE}")
    print(f"  Mode    : risk_metric={RISK_METRIC}")
    print(f"  Uncert. : enabled={STOCHASTIC_EVAL}  scenarios={MC_SCENARIOS}  "
          f"seed={MC_BASE_SEED}")
    print(f"  CCP     : cost={CONFIDENCE_COST}  emission={CONFIDENCE_EMISSION}  "
          f"time={CONFIDENCE_TIME} (independent empirical quantiles)")
    print(f"            mode_cv={MODE_TIME_CV}  mode_cap={MODE_TIME_MAX_FACTOR}")
    print(f"            requested_speed_override_kmh={CLI_MODE_SPEED_OVERRIDE_KMH}  "
          "(resolved against workbook Mode_Speeds once loaded, see [INIT] below)")
    print(f"            border_cv={BORDER_DELAY_CV}  "
          f"border_cap={BORDER_DELAY_MAX_FACTOR}")
    print(f"            border_events={args.border_event_data}")
    print("  Lateness: economic scenario cost only (not a feasibility constraint)")
    print(f"  FeasSeed: fraction={FEASIBILITY_SEED_FRACTION:.2f}  "
          f"restarts={FEASIBILITY_SEARCH_RESTARTS}  "
          f"iterations={FEASIBILITY_SEARCH_ITERATIONS}")
    penalty_label = ("input hourly values" if args.use_input_late_penalties
                     else f"{sourced_late_penalty_h:g} USD/TEU/hour"
                          f" = {sourced_late_penalty_h * 24.0:g}/day")
    print(f"            penalty={penalty_label}")
    print(f"  Emission: payload={PAYLOAD_TONNES_PER_TEU:g} t/TEU  "
          f"wait_override={args.wait_emission_g_per_teu_h}")
    batch_source = args.batches_csv if args.batches_csv else DATA_FILE
    print(f"  Batches : {batch_source}  caps=input data")
    print("=" * 72)

    print("\n[INIT] Loading network data...")
    (node_names, node_region,
     node_hold_cost, node_proc_cost, node_trans_cost,
     arcs, timetables, raw_batches,
     waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
     carbon_tax_map, emission_factor_map, mode_speeds_map,
     trans_map, border_delay_map, theta_rm,
     mode_speed_source) = load_network_from_extended(
        DATA_FILE, cli_speed_override=CLI_MODE_SPEED_OVERRIDE_KMH)

    if args.batches_csv:
        raw_batches = load_batches_from_csv(args.batches_csv)

    input_penalties = [float(batch.penalty_per_teu_h) for batch in raw_batches]
    if args.use_input_late_penalties:
        if input_penalties and max(input_penalties) > 10.0:
            print("[WARN] Input late penalties exceed 10 USD/TEU/hour "
                  f"(range={min(input_penalties):.4g}--{max(input_penalties):.4g}). "
                  "Use only when supported by contract or source data.")
        late_penalty_source = "input_per_batch_hourly"
    else:
        for batch in raw_batches:
            batch.penalty_per_teu_h = sourced_late_penalty_h
        late_penalty_source = "sourced_common_daily_rate_converted_to_hourly"
        if input_penalties:
            print("[INFO] Replaced input late-penalty range "
                  f"{min(input_penalties):.4g}--{max(input_penalties):.4g} "
                  f"USD/TEU/hour with {sourced_late_penalty_h:.4g} "
                  "USD/TEU/hour. Use --use-input-late-penalties only with "
                  "documented contract values.")

    if args.wait_emission_g_per_teu_h is not None:
        wait_emis_g_per_teu_h = max(
            0.0, float(args.wait_emission_g_per_teu_h))
        print("[INFO] Waiting emission overridden to "
              f"{wait_emis_g_per_teu_h:g} gCO2/TEU/hour.")
    elif wait_emis_g_per_teu_h <= 0.0:
        print("[WARN] Waiting emission is zero: the emission CCP is "
              "deterministic under the current system boundary.")
    print_waiting_emission_configuration(wait_emis_g_per_teu_h)

    if args.expected_batches:
        assert len(raw_batches) == args.expected_batches, \
            f"Expected {args.expected_batches} batches, got {len(raw_batches)}"
    print(f"[INIT] Batches confirmed: {len(raw_batches)}")

    print("[INIT] Building path library (shared across all runs)...")
    tt_dict    = build_timetable_dict(timetables)
    arc_lookup = build_arc_lookup(arcs)
    random.seed(0); np.random.seed(0)
    path_lib = build_path_library(
        node_names, node_region, arcs, raw_batches, tt_dict, arc_lookup)
    sanity_check_path_lib(raw_batches, path_lib)

    if RISK_METRIC == "ev":
        scenario_set = build_expected_value_scenario_set(
            arcs=arcs,
            border_delay_map=border_delay_map,
            seed=MC_BASE_SEED,
            border_event_definitions=BORDER_EVENT_DEFINITIONS,
        )
        ACTIVE_SCENARIO_SET = scenario_set
        _PATH_SCENARIO_CACHE = {}
    else:
        scenario_set = configure_scenario_set(
            arcs=arcs,
            border_delay_map=border_delay_map,
            size=MC_SCENARIOS,
            seed=MC_BASE_SEED,
            stochastic=STOCHASTIC_EVAL,
            border_event_definitions=BORDER_EVENT_DEFINITIONS,
        )
    print(f"[INIT] Frozen scenario set: S={scenario_set.size}, "
          f"seed={scenario_set.seed}, stochastic={scenario_set.stochastic}")
    print(f"[INIT] Border-event shocks: "
          f"{len(scenario_set.border_event_mean_h)} events mapped to "
          f"{len(scenario_set.arc_border_event)} network arcs")
    reliable_options = build_reliable_path_options(
        raw_batches, path_lib, tt_dict, trans_map, border_delay_map,
        scenario_set, mode=RISK_METRIC)
    scenario_manifest = build_scenario_manifest(
        risk_metric=RISK_METRIC,
        scenario_set=scenario_set,
        confidence_cost=CONFIDENCE_COST,
        confidence_emission=CONFIDENCE_EMISSION,
        confidence_time=CONFIDENCE_TIME,
        mode_time_cv=MODE_TIME_CV,
        mode_time_cap_factor=MODE_TIME_MAX_FACTOR,
        mode_speed_kmh=mode_speeds_map,
        mode_speed_source=mode_speed_source,
        mode_speed_requested_cli_override_kmh=CLI_MODE_SPEED_OVERRIDE_KMH,
        border_delay_cv=BORDER_DELAY_CV,
        border_delay_cap_factor=BORDER_DELAY_MAX_FACTOR,
        border_event_data_file=str(FSPath(args.border_event_data).resolve()),
        border_event_definitions=BORDER_EVENT_DEFINITIONS,
        late_penalty_source=late_penalty_source,
        late_penalty_input_basis=late_penalty_input_basis,
        late_penalty_baseline_usd_per_teu_h=sourced_late_penalty_h,
        use_input_late_penalties=args.use_input_late_penalties,
        payload_tonnes_per_teu=PAYLOAD_TONNES_PER_TEU,
        wait_emission_g_per_teu_h=wait_emis_g_per_teu_h,
        feasibility_seed_fraction=FEASIBILITY_SEED_FRACTION,
        feasibility_search_restarts=FEASIBILITY_SEARCH_RESTARTS,
        feasibility_search_iterations=FEASIBILITY_SEARCH_ITERATIONS,
    )
    (FSPath(OUTPUT_DIR) / "scenario_manifest.json").write_text(
        _json.dumps(scenario_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    all_front_hist: List = []
    all_run_rows:   List = []
    all_paretos:    List = []
    all_populations: List = []

    _hv_runs, _igd_runs, _sp_runs = [], [], []
    _fr_runs, _frs_runs            = [], []
    _mc_runs, _me_runs, _mt_runs  = [], [], []
    _psz_runs, _fc_runs            = [], []
    _boost_runs                    = []

    total_wall_start = time.perf_counter()

    for run_id in range(RUNS):
        seed = BASE_SEED + run_id
        random.seed(seed); np.random.seed(seed)

        print(f"\n{'='*72}")
        print(f"  RUN {run_id+1:02d}/{RUNS}  |  seed={seed}")
        print(f"{'='*72}")

        (population, pareto,
         front_hist_objs,
         feasible_ratio_hist, feasible_ratio_strict_hist,
         vio_mean_hist,
         boost_trigger_hist, boost_new_feas_hist,
         pareto_size_hist, feasible_count_hist,
         runtime_s) = run_nsga2(
            node_names, node_region, node_hold_cost, node_proc_cost, node_trans_cost,
            arcs, timetables, raw_batches,
            waiting_cost_per_teu_h, wait_emis_g_per_teu_h,
            carbon_tax_map, emission_factor_map, mode_speeds_map,
            trans_map, border_delay_map, theta_rm, path_lib,
            reliable_options,
            pop_size=POP_SIZE, generations=GENERATIONS,
        )

        all_front_hist.append(front_hist_objs)
        all_paretos.append(pareto)
        all_populations.append(population)
        _fr_runs.append(feasible_ratio_hist)
        _frs_runs.append(feasible_ratio_strict_hist)
        _psz_runs.append(pareto_size_hist)
        _fc_runs.append(feasible_count_hist)
        _boost_runs.append(boost_trigger_hist)

        def _bmin_run(idx):
            vals = []
            for gf in front_hist_objs:
                a = _finite_points_array(gf)
                vals.append(float(np.min(a[:, idx])) if a.shape[0] > 0 else np.nan)
            return _ffill_nan(np.array(vals))

        _mc_runs.append(_bmin_run(0))
        _me_runs.append(_bmin_run(1))
        _mt_runs.append(_bmin_run(2))

        best_pen_ind = min(population, key=constraint_sort_key) if population else None
        best_bd = (best_pen_ind.vio_breakdown or {}) if best_pen_ind is not None else {}

        all_run_rows.append({
            "run_id":                   run_id,
            "seed":                     seed,
            "runtime_s":                round(runtime_s, 2),
            "final_pareto_size":        len(pareto),
            "final_FeasRatio_soft":     float(feasible_ratio_hist[-1]),
            "final_FeasRatio_strict":   float(feasible_ratio_strict_hist[-1]),
            "best_penalty":             float(best_pen_ind.penalty) if best_pen_ind else float("inf"),
            "best_normalized_violation": float(best_pen_ind.normalized_violation) if best_pen_ind else float("inf"),
            "best_miss_alloc":          float(best_bd.get("miss_alloc", 0.0)),
            "best_miss_tt":             float(best_bd.get("miss_tt", 0.0)),
            "best_cap_excess":          float(best_bd.get("cap_excess", 0.0)),
            "best_border_cap_excess":   float(best_bd.get("border_cap_excess", 0.0)),
            "best_late_teu_h":          float(best_bd.get("late_teu_h", 0.0)),
            "best_max_observed_late_h": float(best_bd.get("max_observed_late_h", 0.0)),
            "best_min_on_time_prob":    float(best_bd.get("min_on_time_prob", 0.0)),
            "boost_gens_triggered":     int(sum(boost_trigger_hist)),
            "boost_new_feasible_total": int(sum(boost_new_feas_hist)),
        })
        print(f"  [Run {run_id+1:02d}] Pareto={len(pareto)}  "
              f"FeasSoft={feasible_ratio_hist[-1]:.1%}  "
              f"Runtime={runtime_s:.1f}s")

    total_wall = time.perf_counter() - total_wall_start
    print(f"\n[INFO] All {RUNS} runs done in {total_wall:.1f}s")

    # ── Build P* ─────────────────────────────────────────
    P_star = build_P_star_fast(all_front_hist)
    if P_star:
        P_arr  = np.array(P_star, dtype=float)
        mins, maxs = np.min(P_arr, axis=0), np.max(P_arr, axis=0)
    else:
        mins, maxs = np.zeros(NUM_OBJ), np.ones(NUM_OBJ)

    hv_calc = HypervolumeCalculator(ref_point=HV_REF_NORM, num_samples=HV_SAMPLES,
                                     seed=HV_MC_SEED)
    Pn = normalize_points(P_star, mins, maxs) if P_star else []

    for r, front_hist_objs in enumerate(all_front_hist):
        hv_h, igd_h, sp_h = [], [], []
        last_hv = last_igd = last_sp = 0.0
        for gi, gf in enumerate(front_hist_objs):
            if gi % HV_EVERY == 0:
                pts     = [tuple(x) for x in _finite_points_array(gf)]
                An      = clip_points(normalize_points(pts, mins, maxs), HV_REF_NORM) if pts else []
                last_hv = hv_calc.calculate_points(An) if An else 0.0
            if gi % METRIC_EVERY == 0:
                pts      = [tuple(x) for x in _finite_points_array(gf)]
                An       = normalize_points(pts, mins, maxs) if pts else []
                last_igd = igd_plus(Pn, An) if (Pn and An) else float("inf")
                last_sp  = spacing_metric(An) if An else 0.0
            hv_h.append(last_hv)
            igd_h.append(last_igd)
            sp_h.append(last_sp)
        _hv_runs.append(hv_h)
        _igd_runs.append(igd_h)
        _sp_runs.append(sp_h)
        all_run_rows[r]["final_HV_norm"]  = float(hv_h[-1])
        all_run_rows[r]["final_IGD_plus"] = float(igd_h[-1]) if np.isfinite(igd_h[-1]) else None
        all_run_rows[r]["final_Spacing"]  = float(sp_h[-1])

    hv_mat  = np.array(_hv_runs,  dtype=float)
    igd_mat = np.array(_igd_runs, dtype=float)
    sp_mat  = np.array(_sp_runs,  dtype=float)
    fr_mat  = np.array(_fr_runs,  dtype=float)
    frs_mat = np.array(_frs_runs, dtype=float)
    mc_mat  = np.array(_mc_runs,  dtype=float)
    me_mat  = np.array(_me_runs,  dtype=float)
    mt_mat  = np.array(_mt_runs,  dtype=float)
    psz_mat = np.array(_psz_runs, dtype=float)
    fc_mat  = np.array(_fc_runs,  dtype=float)

    gen_arr   = np.arange(GENERATIONS)
    hv_mean,  hv_std  = np.mean(hv_mat,  0), np.std(hv_mat,  0)
    igd_mean, igd_std = np.mean(igd_mat, 0), np.std(igd_mat, 0)
    sp_mean,  sp_std  = np.mean(sp_mat,  0), np.std(sp_mat,  0)
    fr_mean,  fr_std  = np.mean(fr_mat,  0), np.std(fr_mat,  0)
    frs_mean, frs_std = np.mean(frs_mat, 0), np.std(frs_mat, 0)

    best_run_idx  = int(np.argmax(hv_mat[:, -1]))
    best_pareto   = all_paretos[best_run_idx]
    best_boost    = _boost_runs[best_run_idx]
    min_cost_best = mc_mat[best_run_idx]
    min_emis_best = me_mat[best_run_idx]
    min_time_best = mt_mat[best_run_idx]

    mc_mean, mc_std = np.nanmean(mc_mat, 0), np.nanstd(mc_mat, 0)
    me_mean, me_std = np.nanmean(me_mat, 0), np.nanstd(me_mat, 0)
    mt_mean, mt_std = np.nanmean(mt_mat, 0), np.nanstd(mt_mat, 0)

    p = lambda name: str(FSPath(OUTPUT_DIR) / name)

    # ── Per-generation metrics CSV ────────────────────────
    df_metrics = pd.DataFrame({"generation": gen_arr})
    df_metrics["HV_mean"] = hv_mean;  df_metrics["HV_std"] = hv_std
    df_metrics["IGD_plus_mean"] = igd_mean; df_metrics["IGD_plus_std"] = igd_std
    df_metrics["Spacing_mean"] = sp_mean;   df_metrics["Spacing_std"] = sp_std
    df_metrics["FeasRatio_soft_mean"]   = fr_mean;  df_metrics["FeasRatio_soft_std"]   = fr_std
    df_metrics["FeasRatio_strict_mean"] = frs_mean; df_metrics["FeasRatio_strict_std"] = frs_std
    df_metrics["MinCost_best"]  = min_cost_best
    df_metrics["MinEmis_best"]  = min_emis_best
    df_metrics["MinTime_best"]  = min_time_best
    df_metrics["MinCost_mean"]  = mc_mean; df_metrics["MinCost_std"]  = mc_std
    df_metrics["MinEmis_mean"]  = me_mean; df_metrics["MinEmis_std"]  = me_std
    df_metrics["MinTime_mean"]  = mt_mean; df_metrics["MinTime_std"]  = mt_std
    df_metrics["boost_triggered_best"] = best_boost
    df_metrics.to_csv(p("metrics_per_generation.csv"), index=False)
    print(f"[EXPORT] metrics_per_generation.csv saved.")

    # ── Per-run summary Excel ─────────────────────────────
    df_runs = pd.DataFrame(all_run_rows)
    def _s(col):
        v = df_runs[col].dropna().astype(float)
        return f"{v.mean():.4f} ± {v.std():.4f}"
    agg_row = {c: _s(c) if df_runs[c].dtype != object else "—" for c in df_runs.columns}
    agg_row["run_id"] = "mean±std"
    df_runs = pd.concat([df_runs, pd.DataFrame([agg_row])], ignore_index=True)
    xlsx_path = p("run_summary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_runs.to_excel(writer, sheet_name="RunSummary", index=False)
        ws = writer.sheets["RunSummary"]
        try:
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.utils import get_column_letter
            hfill = PatternFill("solid", fgColor="1F4E79")
            hfont = Font(color="FFFFFF", bold=True)
            afill = PatternFill("solid", fgColor="FFF2CC")
            for ci, col in enumerate(df_runs.columns, 1):
                c = ws.cell(row=1, column=ci)
                c.fill, c.font = hfill, hfont
                c.alignment = Alignment(horizontal="center")
                ws.column_dimensions[get_column_letter(ci)].width = min(len(col)+4, 24)
            for ci in range(1, len(df_runs.columns)+1):
                ws.cell(row=len(df_runs)+1, column=ci).fill = afill
            ws.freeze_panes = "A2"
        except Exception:
            pass
    print(f"[EXPORT] run_summary.xlsx → {xlsx_path}")

    # ── Pareto outputs (best run) ─────────────────────────
    save_pareto_solutions(best_pareto, raw_batches, p("result.txt"))
    export_pareto_points_json(best_pareto, raw_batches, out_json=p("pareto_points.json"))
    if not best_pareto and all_populations:
        export_best_infeasible_json(
            all_populations[best_run_idx], raw_batches,
            out_json=p("best_infeasible.json"))

    # ── Front history JSON ────────────────────────────────
    _fh_export = []
    for fh in all_front_hist:
        _fh_export.append([[list(obj) for obj in gf] for gf in fh])
    with open(p("front_hist_uncertainty.json"), "w", encoding="utf-8") as _f:
        _json.dump(_fh_export, _f)
    print(f"[EXPORT] nsga2_front_hist.json saved.")

    # ── Plots ─────────────────────────────────────────────
    ip_m = np.where(np.isfinite(igd_mean), igd_mean, np.nan)
    ip_s = np.where(np.isfinite(igd_std),  igd_std,  np.nan)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen_arr, hv_mean, lw=2.2, color="#9C27B0", label="HV mean")
    _band(ax, gen_arr, hv_mean, hv_std, "#9C27B0")
    ax.plot(gen_arr, hv_mat[best_run_idx], lw=1.4, ls="--",
            color="#E91E63", alpha=0.8, label=f"Best run #{best_run_idx}")
    ax.set_xlabel("Generation"); ax.set_ylabel("HV (normalised)")
    ax.set_ylim(bottom=0); ax.set_title(f"NSGA-II Hypervolume — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_HV.png")); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen_arr, ip_m, lw=2.2, color="#E53935", label="IGD+ mean")
    ax.fill_between(gen_arr,
                    np.where(np.isfinite(ip_m - ip_s), ip_m - ip_s, np.nan),
                    np.where(np.isfinite(ip_m + ip_s), ip_m + ip_s, np.nan),
                    alpha=0.18, color="#E53935")
    ax.set_xlabel("Generation"); ax.set_ylabel("IGD+ (lower is better)")
    ax.set_title(f"NSGA-II IGD+ — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_IGDplus.png")); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen_arr, sp_mean, lw=2.2, color="#00897B", label="Spacing mean")
    _band(ax, gen_arr, sp_mean, sp_std, "#00897B")
    ax.set_xlabel("Generation"); ax.set_ylabel("Spacing")
    ax.set_title(f"NSGA-II Spacing — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_Spacing.png")); plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 4), dpi=180)
    ax1.plot(gen_arr, hv_mean, lw=2.0, color="#9C27B0", label="HV mean")
    _band(ax1, gen_arr, hv_mean, hv_std, "#9C27B0")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("HV (normalised)", color="#9C27B0")
    ax1.tick_params(axis="y", labelcolor="#9C27B0"); ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(gen_arr, ip_m, lw=2.0, ls="--", color="#E53935", label="IGD+ mean")
    ax2.set_ylabel("IGD+", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="lower right", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)
    plt.title(f"NSGA-II Convergence: HV & IGD+ ({RUNS} runs)")
    plt.tight_layout(); plt.savefig(p("plot_Convergence.png")); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4), dpi=180)
    ax.plot(gen_arr, fr_mean,  lw=2.2, color="#1976D2", label="Soft mean")
    _band(ax, gen_arr, fr_mean, fr_std, "#1976D2")
    ax.plot(gen_arr, frs_mean, lw=2.0, ls="--", color="#FB8C00", label="Strict mean")
    _band(ax, gen_arr, frs_mean, frs_std, "#FB8C00", alpha=0.13)
    bgs = [g for g, v in enumerate(best_boost) if v > 0]
    if bgs:
        ax.scatter(bgs, [fr_mean[g] for g in bgs if g < len(fr_mean)],
                   marker="^", color="red", s=40, zorder=5,
                   label=f"Boost best run ({len(bgs)})")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Generation"); ax.set_ylabel("Feasible Ratio")
    ax.set_title(f"NSGA-II Feasible Ratio — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_FeasibleRatio.png")); plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), dpi=180, sharex=True)
    fig.subplots_adjust(hspace=0.35)
    for ax, (mean, std, best, ylabel, color) in zip(axes, [
        (mc_mean, mc_std, min_cost_best, "Min Cost ($)",       "#E53935"),
        (me_mean, me_std, min_emis_best, "Min Emission (gCO₂)","#00897B"),
        (mt_mean, mt_std, min_time_best, "Min Time (h)",        "#FB8C00"),
    ]):
        ax.plot(gen_arr, mean, lw=2.0, color=color, label="Mean")
        _band(ax, gen_arr, mean, std, color)
        ax.plot(gen_arr, best, lw=1.4, ls="--", color="black",
                alpha=0.7, label=f"Best run #{best_run_idx}")
        mask = np.isfinite(best)
        if mask.any():
            fg = int(np.where(mask)[0][0])
            ax.axvline(fg, color="green", ls=":", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=9); ax.legend(fontsize=7)
        ax.grid(True, ls=":", alpha=0.5)
    axes[-1].set_xlabel("Generation")
    fig.suptitle(f"NSGA-II Min Objectives per Generation ({RUNS} runs, mean±std)",
                 fontsize=12, fontweight="bold")
    plt.savefig(p("plot_MinObjectives.png")); plt.close()

    best_pts = unique_objective_tuples(
        [ind.objectives for ind in best_pareto if ind.feasible], tol=1e-9)
    plot_pareto_3d(best_pts, save=p("plot_Pareto3D.png"),
                   title=f"NSGA-II Pareto — Best Run #{best_run_idx}")
    plot_pareto_2d(best_pts, save=p("plot_Pareto2D.png"))

    all_pts_by_run = [[ind.objectives for ind in run_p if ind.feasible]
                      for run_p in all_paretos]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    pairs = [(0,1,"Cost ($)","Emission (gCO₂)"),
             (0,2,"Cost ($)","Time (h)"),
             (1,2,"Emission (gCO₂)","Time (h)")]
    for ax, (xi, yi, xl, yl) in zip(axes, pairs):
        for r_idx, pts in enumerate(all_pts_by_run):
            arr = _finite_points_array(pts)
            if arr.shape[0] == 0: continue
            if r_idx == best_run_idx: continue
            ax.scatter(arr[:, xi], arr[:, yi], s=10, alpha=0.3, color="#90CAF9")
        ab = _finite_points_array(all_pts_by_run[best_run_idx])
        if ab.shape[0] > 0:
            sc = ax.scatter(ab[:, xi], ab[:, yi], s=28, alpha=0.95,
                            c=ab[:, 0], cmap="viridis", zorder=5,
                            label=f"Best run #{best_run_idx}")
            plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Cost ($)")
            ax.legend(fontsize=7)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"{xl} vs {yl}")
        ax.grid(True, ls=":", alpha=0.45)
    fig.suptitle(f"NSGA-II Pareto Front 2D — All {RUNS} Runs (grey) + Best Run (colour)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig(p("plot_Pareto2D_allruns.png")); plt.close()

    psz_mean = np.mean(psz_mat, 0); psz_std = np.std(psz_mat, 0)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    ax.plot(gen_arr, psz_mean, lw=2.2, color="#5C6BC0", label="Mean")
    _band(ax, gen_arr, psz_mean, psz_std, "#5C6BC0")
    ax.plot(gen_arr, psz_mat[best_run_idx], lw=1.4, ls="--",
            color="#E91E63", alpha=0.8, label=f"Best run #{best_run_idx}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Non-dominated feasible size")
    ax.set_title(f"NSGA-II Pareto Size — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_ParetoSize.png")); plt.close()

    fc_mean = np.mean(fc_mat, 0); fc_std = np.std(fc_mat, 0)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    ax.plot(gen_arr, fc_mean, lw=2.2, color="#1976D2", label="Mean")
    _band(ax, gen_arr, fc_mean, fc_std, "#1976D2")
    ax.axhline(POP_SIZE, color="grey", ls="--", lw=1.0, label=f"Pop={POP_SIZE}")
    ax.set_xlabel("Generation"); ax.set_ylabel("Feasible solutions in population")
    ax.set_title(f"NSGA-II Feasible Count — mean±std ({RUNS} runs)")
    ax.legend(fontsize=8); ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_FeasibleCount.png")); plt.close()

    rts = [r["runtime_s"] for r in all_run_rows]
    avg_rt = float(np.mean(rts))
    colors_rt = ["#E53935" if r > avg_rt else "#1976D2" for r in rts]
    fig, ax = plt.subplots(figsize=(max(8, RUNS*0.4), 4), dpi=180)
    ax.bar(range(RUNS), rts, color=colors_rt, alpha=0.85)
    ax.axhline(avg_rt, color="black", ls="--", lw=1.5, label=f"Mean={avg_rt:.1f}s")
    ax.set_xlabel("Run ID"); ax.set_ylabel("Runtime (s)")
    ax.set_title("NSGA-II Runtime per Run"); ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout(); plt.savefig(p("plot_Runtime.png")); plt.close()

    final_hv  = hv_mat[:,  -1]
    final_igd = igd_mat[:, -1]
    final_sp  = sp_mat[:,  -1]
    final_fr  = fr_mat[:,  -1]
    final_frs = frs_mat[:, -1]
    table_data = []
    for name, vals in [
        ("HV_norm (↑)",          final_hv),
        ("IGD+ (↓)",             final_igd),
        ("Spacing (↓)",          final_sp),
        ("FeasRatio soft (↑)",   final_fr),
        ("FeasRatio strict (↑)", final_frs),
        ("Runtime (s)",          np.array(rts)),
    ]:
        f = vals[np.isfinite(vals)]
        if len(f) == 0:
            table_data.append([name, "—", "—", "—", "—"])
        else:
            table_data.append([name,
                                f"{np.min(f):.4f}", f"{np.max(f):.4f}",
                                f"{np.mean(f):.4f}", f"{np.std(f):.4f}"])
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    ax.axis("off")
    tbl = ax.table(cellText=table_data,
                   colLabels=["Metric","Min","Max","Mean","Std"],
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for j in range(5):
        tbl[(0,j)].set_facecolor("#1F4E79")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        clr = "#EBF3FB" if i % 2 == 0 else "#FFFFFF"
        for j in range(5): tbl[(i+1,j)].set_facecolor(clr)
    fig.suptitle(f"NSGA-II Tuned Baseline — {RUNS} Runs (Final Generation)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(p("plot_SummaryTable.png"),
                                    bbox_inches="tight"); plt.close()

    print("\n" + "=" * 72)
    print(f"  NSGA-II STOCHASTIC EXTENSION  {RUNS} RUNS COMPLETE")
    print(f"  Total wall time:    {total_wall:.1f}s  ({total_wall/60:.1f} min)")
    print(f"  Best run:           #{best_run_idx}  (highest final HV)")
    print(f"  Pareto size (best): {len(best_pareto)}")
    print(f"  HV   mean±std:      {np.mean(final_hv):.4f} ± {np.std(final_hv):.4f}")
    fin_igd = final_igd[np.isfinite(final_igd)]
    if len(fin_igd):
        print(f"  IGD+ mean±std:      {np.mean(fin_igd):.4f} ± {np.std(fin_igd):.4f}")
    print(f"  Spacing mean±std:   {np.mean(final_sp):.4f} ± {np.std(final_sp):.4f}")
    print(f"  FeasSoft mean±std:  {np.mean(final_fr):.1%} ± {np.std(final_fr):.1%}")
    print(f"  Runtime mean±std:   {np.mean(rts):.1f}s ± {np.std(rts):.1f}s")
    print(f"  Outputs in:         {OUTPUT_DIR}/")
    print("=" * 72)
