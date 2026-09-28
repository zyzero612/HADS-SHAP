from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from hads_shap.core import cluster_features, dual_permutation_test
from hads_shap.gas import fit_xgboost, load_dataset, stratified_holdout
from hads_shap.reporting import state_summary


def _balanced_audit_indices(y, batch, indices, per_stratum=10, seed=20260926):
    rng = np.random.default_rng(seed)
    selected = []
    for b in np.unique(batch):
        for c in np.unique(y):
            idx = indices[(batch[indices] == b) & (y[indices] == c)]
            if len(idx):
                selected.extend(rng.choice(idx, size=min(per_stratum, len(idx)), replace=False))
    return np.asarray(sorted(selected))


def _magnitude(values, n_features):
    values = np.asarray(values)
    if values.ndim == 2:
        return np.abs(values)
    if values.ndim == 3 and values.shape[1] == n_features:
        return np.abs(values).mean(axis=2)
    if values.ndim == 3 and values.shape[2] == n_features:
        return np.abs(values).mean(axis=1)
    raise RuntimeError(f"Unexpected SHAP shape: {values.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/gas_sensor_array_drift.zip"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, default=Path("results/generated/shap_dependence"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    X, y, batch = load_dataset(args.data)
    train, holdout = stratified_holdout(y, batch, seed=args.seed)
    audit = _balanced_audit_indices(y, batch, holdout, per_stratum=10, seed=args.seed)
    model = fit_xgboost(X[train], y[train], seed=args.seed)
    groups, _ = cluster_features(pd.DataFrame(X[train]), threshold=0.90)

    background_rng = np.random.default_rng(args.seed)
    background = X[background_rng.choice(train, size=min(100, len(train)), replace=False)]

    settings = {
        "tree_path": shap.TreeExplainer(model, feature_perturbation="tree_path_dependent"),
        "interventional": shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
        ),
    }

    rows = []
    for name, explainer in settings.items():
        values = explainer(X[audit], check_additivity=False).values
        A = _magnitude(values, X.shape[1])
        result = dual_permutation_test(
            A,
            batch[audit],
            groups,
            permutations=args.permutations,
            strata=y[audit],
            seed=args.seed,
        )
        result.table.to_csv(args.out / f"{name}_groups.csv", index=False)
        rows.append({"setting": name, **state_summary(result.table)})

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
