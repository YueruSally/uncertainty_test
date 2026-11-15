#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/expanda"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
workers="${MAX_WORKERS:-2}"
python_bin="${PYTHON:-python3}"
case "${1:-help}" in
  collect)     set -- component_screen_b20_v1/run.py --stage collect --workers "$workers" ;;
  shared)      set -- joint_icm_multiscale_v1/run.py --stage all --workers "$workers" ;;
  independent) set -- joint_icm_multiscale_terminal_v1/run.py --stage independent --workers "$workers" ;;
  compare)     set -- component_joint_b5000_v1/run.py --stage all --workers "$workers" ;;
  ppo)         set -- ppo_joint_b5000_pilot_v2/run.py --workers "$workers" ;;
  confirm)     set -- ppo_joint_b5000_screen27_v1/run.py --workers "$workers" ;;
  export)      set -- paper_evidence_b5000_v1/collect.py ;;
  help|-h|--help) echo 'Stages in execution order: collect, shared, independent, compare, ppo, confirm, export'; exit 0 ;;
  *) echo "Unknown stage: $1" >&2; exit 2 ;;
esac
exec "$python_bin" -u "$@"
