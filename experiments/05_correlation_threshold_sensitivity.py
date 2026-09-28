from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hads_shap.beijing import run_xgboost_target
from hads_shap.gas import run_xgboost
from hads_shap.reporting import state_summary


THRESHOLDS = (0.80, 0.85, 0.90, 0.95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beijing-data",
        type=Path,
        default=Path("data/raw/beijing_multi_site_air_quality.zip"),
    )
    parser.add_argument(
        "--gas-data",
        type=Path,
        default=Path("data/raw/gas_sensor_array_drift.zip"),
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/threshold_sensitivity"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    beijing_rows, gas_rows = [], []

    for threshold in THRESHOLDS:
        _, bj = run_xgboost_target(
            args.beijing_data,
            "Aotizhongxin",
            horizon=24,
            threshold=threshold,
            permutations=args.permutations,
            seed=args.seed,
        )
        beijing_rows.append({"threshold": threshold, **state_summary(bj.table)})

        _, gas, _, _ = run_xgboost(
            args.gas_data,
            threshold=threshold,
            permutations=args.permutations,
            seed=args.seed,
        )
        gas_rows.append({"threshold": threshold, **state_summary(gas.table)})

    pd.DataFrame(beijing_rows).to_csv(args.out / "beijing.csv", index=False)
    pd.DataFrame(gas_rows).to_csv(args.out / "gas.csv", index=False)
    print(pd.DataFrame(beijing_rows).to_string(index=False))
    print(pd.DataFrame(gas_rows).to_string(index=False))


if __name__ == "__main__":
    main()
