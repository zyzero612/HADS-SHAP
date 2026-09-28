from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from hads_shap.core import cluster_features, dual_permutation_test
from hads_shap.gas import load_dataset, stratified_holdout
from hads_shap.reporting import state_summary


def _tree_shap_magnitude(model, X):
    values = shap.TreeExplainer(model)(X, check_additivity=False).values
    values = np.asarray(values)
    if values.ndim == 2:
        return np.abs(values)
    if values.ndim == 3 and values.shape[1] == X.shape[1]:
        return np.abs(values).mean(axis=2)
    if values.ndim == 3 and values.shape[2] == X.shape[1]:
        return np.abs(values).mean(axis=1)
    raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")


def _performance(model, X, y, batch, holdout):
    prediction = model.predict(X[holdout])
    rows = []
    for b in np.unique(batch):
        mask = batch[holdout] == b
        rows.append(
            {
                "batch": int(b),
                "n": int(mask.sum()),
                "balanced_accuracy": balanced_accuracy_score(
                    y[holdout][mask], prediction[mask]
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/gas_sensor_array_drift.zip"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/gas_models"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    X, y, batch = load_dataset(args.data)
    train, holdout = stratified_holdout(y, batch, seed=args.seed)
    groups, _ = cluster_features(pd.DataFrame(X[train]), threshold=args.threshold)

    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=args.seed,
            n_jobs=8,
            verbosity=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=30,
            random_state=args.seed,
            n_jobs=8,
            max_features="sqrt",
        ),
    }

    summaries = []
    for name, model in models.items():
        model.fit(X[train], y[train])
        perf = _performance(model, X, y, batch, holdout)
        A = _tree_shap_magnitude(model, X[holdout])
        result = dual_permutation_test(
            A,
            batch[holdout],
            groups,
            permutations=args.permutations,
            strata=y[holdout],
            seed=args.seed,
        )
        perf.to_csv(args.out / f"{name}_performance.csv", index=False)
        result.table.to_csv(args.out / f"{name}_groups.csv", index=False)
        summaries.append(
            {
                "model": name,
                "mean_balanced_accuracy": perf.balanced_accuracy.mean(),
                **state_summary(result.table),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
