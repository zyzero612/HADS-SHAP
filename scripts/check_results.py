"""Summarize stored per-run and per-group results used in the manuscript."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "results" / "reference"
STATES = ("Stable", "Mass-only", "WGAM-only", "Mixed")


def main() -> None:
    runs = pd.read_csv(ROOT / "synthetic" / "synthetic_ground_truth_100rep.csv")
    confusion = pd.crosstab(runs.state, runs.pred).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    confusion.columns.name = None
    recorded = pd.read_csv(ROOT / "synthetic" / "confusion_matrix.csv").set_index("state")
    assert len(runs) == 400
    pd.testing.assert_frame_equal(confusion, recorded.loc[list(STATES), list(STATES)])
    print("Controlled experiment (Table 2):")
    print(confusion.to_string())

    groups = [
        pd.read_csv(ROOT / "beijing" / f"{target}_groups.csv")
        for target in ("Aotizhongxin", "Changping", "Dingling")
    ]
    counts = pd.concat(groups).state.value_counts().reindex(STATES, fill_value=0)
    assert len(pd.concat(groups)) == 252
    print("\nBeijing groups (Table 4):")
    print(counts.to_string())

    gas = pd.read_csv(ROOT / "gas" / "xgb_groups.csv")
    assert len(gas) == 25
    print("\nGas Sensor XGBoost groups (Section 5.3):")
    print(gas.state.value_counts().to_string())


if __name__ == "__main__":
    main()
