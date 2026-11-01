#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
SMOKE_ROOT=${SMOKE_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_balanced_teacher_transformer_v2_smoke_1inst_1seed_b260}
DATA_ROOT=${DATA_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_balanced_teacher_transformer_v2_data_7inst_2seed_b400}
TRAIN_ROOT=${TRAIN_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_balanced_teacher_transformer_v2_train}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_balanced_teacher_transformer_v2_pilot_3inst_3seed_b400}

for REQUIRED in "$NORMALIZATION" "$BOOTSTRAP_CHECKPOINT"; do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$SMOKE_ROOT" "$DATA_ROOT" "$TRAIN_ROOT" "$OUT_ROOT"

echo "===== TEST M TRANSFORMER TEACHER PIPELINE ====="
python3 -u -m unittest \
  tests.test_m_balanced_teacher_transformer \
  tests.test_learning_components_v2_controls.V2ControlTests.test_m_scheduled_operator_counts_base_before_start_generation \
  tests.test_learning_components_v2_controls.V2ControlTests.test_forced_replace_model_selector_executes_transformer_candidate \
  > "$OUT_ROOT/tests.log" 2>&1

echo "===== REAL 1-INSTANCE / 1-SEED ROLLOUT SMOKE ====="
python3 -u collect_m_teacher_rollouts.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --out "$SMOKE_ROOT" \
  --instance-ids train-S0-002 \
  --algorithm-seeds 880001 \
  --minimum-generation 10 \
  --max-workers 1 \
  --evaluation-budget 260 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$SMOKE_ROOT/collection.log"

if ! jq -e '
  .completed == true
  and .run_count == 1
  and .policy_state_count > 0
' "$SMOKE_ROOT/DATA_COMPLETE.json" >/dev/null; then
  echo "M TRANSFORMER REAL ROLLOUT SMOKE FAILED" >&2
  exit 1
fi

echo "===== COLLECT 7-INSTANCE BALANCED-TEACHER STATES ====="
python3 -u collect_m_teacher_rollouts.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --out "$DATA_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 \
    train-S0-008 train-S0-009 train-S0-011 \
  --algorithm-seeds 881001 881002 \
  --minimum-generation 10 \
  --max-workers 7 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$DATA_ROOT/collection.log"

echo "===== DISTIL BALANCED TEACHER INTO PATH-V4 TRANSFORMER ====="
python3 -u train_m_balanced_teacher_transformer.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --out "$TRAIN_ROOT" \
  --validation-instances S0_011 \
  --epochs 20 \
  --batch-size 1 \
  --early-stopping-patience 4 \
  --learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --seed 62001 \
  --device cpu \
  | tee "$TRAIN_ROOT/training.log"

echo "===== HELD-OUT 3-INSTANCE / 3-SEED CCP PILOT ====="
python3 -u evaluate_m_balanced_teacher_transformer.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --bootstrap-checkpoint "$BOOTSTRAP_CHECKPOINT" \
  --transformer-checkpoint "$TRAIN_ROOT/final_policy.pt" \
  --training-report "$TRAIN_ROOT/TRAINING_COMPLETE.json" \
  --out "$OUT_ROOT" \
  --instance-ids train-S0-010 train-S0-012 train-S0-013 \
  --algorithm-seeds 882001 882002 882003 \
  --minimum-generation 10 \
  --max-workers 9 \
  --evaluation-budget 400 \
  --scenario-count 200 \
  --ccp-confidence 0.90 \
  --pop 20 \
  --gens 1000 \
  --device cpu \
  | tee "$OUT_ROOT/evaluation.log"

if ! jq -e '
  .completed == true
  and .instance_count == 3
  and .algorithm_seed_count == 3
  and .run_count == 27
  and .scenario_exact == true
  and (.strategies | length) == 3
' "$OUT_ROOT/QUICK_RESULT.json" >/dev/null; then
  echo "M TRANSFORMER PILOT FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M BALANCED-TEACHER TRANSFORMER PILOT COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
