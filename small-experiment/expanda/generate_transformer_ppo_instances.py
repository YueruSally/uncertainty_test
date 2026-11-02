#!/usr/bin/env python3
"""Generate frozen Transformer-PPO train/test instance manifests."""
import argparse
from pathlib import Path

from transformer_ppo.instance_catalog import generate_catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data/data_expanded.xlsx")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-time-max-h", type=float, required=True,
                        help="Scientifically approved upper bound; no implicit default.")
    parser.add_argument("--release-time-step-h", type=float, required=True,
                        help="Scientifically approved release-time grid step.")
    parser.add_argument("--master-seed", type=int, default=31_000_000)
    parser.add_argument("--train-per-config", type=int, default=50)
    parser.add_argument("--test-per-config", type=int, default=10)
    args = parser.parse_args()
    result = generate_catalog(
        args.data, args.out, args.release_time_max_h, args.release_time_step_h,
        master_seed=args.master_seed, train_per_config=args.train_per_config,
        test_per_config=args.test_per_config)
    print({key: result[key] for key in
           ("train_count", "test_count", "catalog_digest")})
