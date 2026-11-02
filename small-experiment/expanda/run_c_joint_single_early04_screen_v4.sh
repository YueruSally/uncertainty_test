#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_c_joint_single_early04_screen_v4_10inst_3seed_b400}

test -f "$NORMALIZATION" || { echo "MISSING REQUIRED INPUT: $NORMALIZATION" >&2; exit 1; }

mkdir -p "$OUT_ROOT"

echo "===== 10-INSTANCE / 3-SEED EARLY-04 JOINT-SINGLE C SCREEN ====="
python3 -u evaluate_c_directed_crossover.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
  --strategies \
    BASE \
    C_JOINT_BALANCED_SINGLE_EARLY04 \
    C_JOINT_BALANCED_SAFE_SINGLE_EARLY04 \
  --max-workers 12 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 10
  and .algorithm_seed_count == 3
  and .run_count == 90
  and ([.status_counts[]] | add) == 90
  and .scenario_exact == true
  and .single_override_exact == true
  and (.strategies | length) == 3
  and (
    [
      .strategies[]
      | select(.strategy != "BASE")
      | .maximum_generation
    ]
    | all(. == 4)
  )
  and (
    [
      .strategies[]
      | select(.strategy != "BASE")
      | .single_override_violations
    ]
    | all(. == 0)
  )
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "C EARLY-04 JOINT-SINGLE SCREEN FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== C EARLY-04 JOINT-SINGLE SCREEN COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
