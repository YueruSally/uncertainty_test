#!/usr/bin/env bash
set -euo pipefail

INSTANCES=${INSTANCES:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/instances}
NORMALIZATION=${NORMALIZATION:-/home/lunet/cokt2/uncertainty-transformer-ppo/expanda/transformer_ppo_runs/reward_normalization.json}
CHECKPOINT=${CHECKPOINT:-transformer_ppo_runs/diagnostics/v5r_joint_replace_smoke_b160_p20_seed9501/M/final_policy.pt}
OUT_ROOT=${OUT_ROOT:-transformer_ppo_runs/diagnostics/ccp_m_directed_balanced_validation_v1_10inst_10seed_b400}

for REQUIRED in \
  "$NORMALIZATION" \
  "$CHECKPOINT"
do
  if ! test -f "$REQUIRED"; then
    echo "MISSING REQUIRED INPUT: $REQUIRED" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

echo "===== FORMAL 10-INSTANCE 10-SEED DIRECTED BALANCED VALIDATION ====="
python3 -u evaluate_m_late_replace_frequency.py \
  --instances "$INSTANCES" \
  --normalization "$NORMALIZATION" \
  --checkpoint "$CHECKPOINT" \
  --strategies BASE REPLACE_G10_JOINT_BALANCED \
  --out "$OUT_ROOT" \
  --instance-ids \
    train-S0-002 train-S0-005 train-S0-006 train-S0-007 train-S0-008 \
    train-S0-009 train-S0-010 train-S0-011 train-S0-012 train-S0-013 \
  --algorithm-seeds \
    777001 777002 777003 777004 777005 \
    777006 777007 777008 777009 777010 \
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
  and .algorithm_seed_count == 10
  and .run_count == 200
  and ([.status_counts[]] | add) == 200
  and .scenario_exact == true
  and (.strategies | length) == 2
  and (.per_instance | length) == 10
  and (.directed_replace_diagnosis.best_strategy
       == "REPLACE_G10_JOINT_BALANCED")
' "$OUT_ROOT/COMPLETE.json" >/dev/null; then
  echo "M DIRECTED BALANCED VALIDATION FAILED COMPLETENESS CHECK" >&2
  exit 1
fi

jq '
  [
    .runs[]
    | select(
        .strategy == "REPLACE_G10_JOINT_BALANCED"
        and .algorithm_seed >= 777004
      )
  ] as $rows
  | ($rows | length) as $n
  | ($rows | map(.paired_percent)) as $values
  | ($values | add / $n) as $mean
  | (
      $values
      | map((. - $mean) * (. - $mean))
      | add / ($n - 1)
      | sqrt
    ) as $sd
  | {
      definition:
        "held-out seeds 777004-777010; these seeds did not select the strategy",
      strategy: "REPLACE_G10_JOINT_BALANCED",
      instance_count: ($rows | map(.instance) | unique | length),
      holdout_seed_count: ($rows | map(.algorithm_seed) | unique | length),
      run_count: $n,
      mean_paired_percent: $mean,
      standard_deviation_paired_percent: $sd,
      standard_error_paired_percent: ($sd / ($n | sqrt)),
      mean_95ci_low: ($mean - 1.96 * $sd / ($n | sqrt)),
      mean_95ci_high: ($mean + 1.96 * $sd / ($n | sqrt)),
      median_paired_percent: ($values | sort | .[$n / 2 | floor]),
      minimum_paired_percent: ($values | min),
      maximum_paired_percent: ($values | max),
      wins: ($values | map(select(. > 1e-12)) | length),
      ties: ($values | map(select((. | fabs) <= 1e-12)) | length),
      losses: ($values | map(select(. < -1e-12)) | length),
      runs_above_5_percent: ($values | map(select(. >= 5)) | length),
      runs_above_10_percent: ($values | map(select(. >= 10)) | length),
      per_instance:
        (
          $rows
          | sort_by(.instance)
          | group_by(.instance)
          | map(
              . as $instance_rows
              | ($instance_rows | map(.paired_percent)) as $instance_values
              | {
                  instance: $instance_rows[0].instance,
                  runs: ($instance_rows | length),
                  mean_paired_percent:
                    ($instance_values | add / length),
                  minimum_paired_percent: ($instance_values | min),
                  maximum_paired_percent: ($instance_values | max),
                  wins:
                    ($instance_values | map(select(. > 1e-12)) | length),
                  losses:
                    ($instance_values | map(select(. < -1e-12)) | length)
                }
            )
        )
    }
' "$OUT_ROOT/COMPLETE.json" > "$OUT_ROOT/HOLDOUT_RESULT.json"

if ! jq -e '
  .instance_count == 10
  and .holdout_seed_count == 7
  and .run_count == 70
' "$OUT_ROOT/HOLDOUT_RESULT.json" >/dev/null; then
  echo "M DIRECTED BALANCED HOLDOUT SUMMARY FAILED" >&2
  exit 1
fi

touch "$OUT_ROOT/PIPELINE_COMPLETE"

echo "===== M DIRECTED BALANCED VALIDATION COMPLETE ====="
cat "$OUT_ROOT/QUICK_RESULT.json"
echo "===== HELD-OUT SEEDS 777004-777010 ====="
cat "$OUT_ROOT/HOLDOUT_RESULT.json"
