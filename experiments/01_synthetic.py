from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hads_shap.synthetic import STATES, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/synthetic"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(
        run_experiment(
            repetitions=args.repetitions,
            permutations=args.permutations,
            seed=args.seed,
        )
    )
    runs.to_csv(args.out / "runs.csv", index=False)

    confusion = pd.crosstab(runs["state"], runs["pred"]).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    confusion.to_csv(args.out / "confusion_matrix.csv")

    summary = (
        runs.assign(correct=runs.state == runs.pred)
        .groupby("state", sort=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            median_p_mass=("p_mass", "median"),
            median_p_alloc=("p_alloc", "median"),
            median_mass_ratio=("mass_ratio", "median"),
            median_alloc_ratio=("alloc_ratio", "median"),
        )
        .reset_index()
    )
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
