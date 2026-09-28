from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hads_shap.beijing import TARGETS, run_xgboost_target
from hads_shap.reporting import state_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/beijing_multi_site_air_quality.zip"),
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/beijing_horizons"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for station in TARGETS:
        for horizon in (6, 12, 24):
            performance, result = run_xgboost_target(
                args.data,
                station,
                horizon=horizon,
                threshold=args.threshold,
                permutations=args.permutations,
                seed=args.seed,
            )
            tag = f"{station}_h{horizon}"
            result.table.to_csv(args.out / f"{tag}_groups.csv", index=False)
            row = {
                "Target": station,
                "Horizon (h)": horizon,
                "R2": performance["r2"],
                "RMSE": performance["rmse"],
                **state_summary(result.table),
            }
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "horizon_full.csv", index=False)
    summary = (
        table.groupby("Horizon (h)")
        .agg(
            R2=("R2", "mean"),
            any_wgam_fraction=("any_wgam_fraction", "mean"),
            wgam_only_fraction=("wgam_only_fraction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.out / "horizon_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
