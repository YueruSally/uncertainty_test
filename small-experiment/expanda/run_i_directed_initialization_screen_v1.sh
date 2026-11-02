#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_i_directed_diverse_screen_v1_10inst_3seed_b400}

test -f "$NORMALIZATION" || { echo "MISSING REQUIRED INPUT: $NORMALIZATION" >&2; exit 1; }

mkdir -p "$OUT_ROOT"

echo "===== 10-INSTANCE / 3-SEED DIRECTED I SCREEN ====="
python3 -u evaluate_i_directed_initialization.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds 777001 777002 777003 \
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
  and .run_count == 120
  and ([.status_counts[]] | add) == 120
  and .scenario_exact == true
  and (.strategies | length) == 4
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "I DIRECTED INITIALIZATION SCREEN FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== I DIRECTED INITIALIZATION SCREEN COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
