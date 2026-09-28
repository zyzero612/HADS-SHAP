from __future__ import annotations

import argparse
from pathlib import Path

from hads_shap.gas import run_xgboost
from hads_shap.reporting import state_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/gas_sensor_array_drift.zip"),
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/gas_main"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    performance, result, _, _ = run_xgboost(
        args.data,
        threshold=args.threshold,
        permutations=args.permutations,
        seed=args.seed,
    )
    performance.to_csv(args.out / "performance.csv", index=False)
    result.table.to_csv(args.out / "groups.csv", index=False)
    print(performance.to_string(index=False))
    print(state_summary(result.table))


if __name__ == "__main__":
    main()
