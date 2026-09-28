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
    parser.add_argument("--out", type=Path, default=Path("results/generated/beijing_main"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for station in TARGETS:
        performance, result = run_xgboost_target(
            args.data,
            station,
            horizon=24,
            threshold=args.threshold,
            permutations=args.permutations,
            seed=args.seed,
        )
        result.table.to_csv(args.out / f"{station}_groups.csv", index=False)
        pd.DataFrame([performance]).to_csv(
            args.out / f"{station}_performance.csv", index=False
        )
        row = {"Target": station, **state_summary(result.table)}
        row["excluded_zero_mass_groups"] = result.excluded_groups
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
