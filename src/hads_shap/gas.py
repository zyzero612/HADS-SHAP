from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import balanced_accuracy_score

from .core import DEFAULT_SEED, cluster_features, dual_permutation_test


def load_dataset(zip_path: str | Path):
    X_all, y_all, batch_all = [], [], []
    with zipfile.ZipFile(Path(zip_path)) as archive:
        for batch in range(1, 11):
            raw = archive.read(f"Dataset/batch{batch}.dat")
            X, y = load_svmlight_file(io.BytesIO(raw), n_features=128)
            X = X.toarray().astype(np.float32)
            y = y.astype(int)
            keep = y <= 5
            X_all.append(X[keep])
            y_all.append(y[keep])
            batch_all.append(np.full(keep.sum(), batch))
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(batch_all)


def stratified_holdout(y, batch, train_fraction=0.70, seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    train_idx, holdout_idx = [], []
    for b in np.unique(batch):
        for c in np.unique(y):
            idx = np.flatnonzero((batch == b) & (y == c))
            idx = rng.permutation(idx)
            n_train = max(1, int(np.floor(train_fraction * len(idx))))
            train_idx.extend(idx[:n_train])
            holdout_idx.extend(idx[n_train:])
    return np.asarray(sorted(train_idx)), np.asarray(sorted(holdout_idx))


def fit_xgboost(X_train, y_train, seed: int = DEFAULT_SEED):
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=8,
    )
    model.fit(X_train, y_train - 1)
    return model


def xgboost_attributions(model, X):
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    if contributions.ndim != 3:
        raise RuntimeError(f"Unexpected XGBoost contribution shape: {contributions.shape}")
    return np.abs(contributions[:, :, :-1]).mean(axis=1)


def run_xgboost(
    zip_path: str | Path,
    threshold: float = 0.90,
    permutations: int = 1000,
    seed: int = DEFAULT_SEED,
):
    X, y, batch = load_dataset(zip_path)
    train_idx, holdout_idx = stratified_holdout(y, batch, seed=seed)
    model = fit_xgboost(X[train_idx], y[train_idx], seed=seed)

    prediction = model.predict(X[holdout_idx]) + 1
    performance = []
    for b in np.unique(batch):
        mask = batch[holdout_idx] == b
        performance.append(
            {
                "batch": int(b),
                "n": int(mask.sum()),
                "balanced_accuracy": balanced_accuracy_score(
                    y[holdout_idx][mask], prediction[mask]
                ),
            }
        )

    attributions = xgboost_attributions(model, X[holdout_idx])
    groups, _ = cluster_features(pd.DataFrame(X[train_idx]), threshold=threshold)
    result = dual_permutation_test(
        attributions,
        batch[holdout_idx],
        groups,
        permutations=permutations,
        strata=y[holdout_idx],
        seed=seed,
    )
    return pd.DataFrame(performance), result, model, (X, y, batch, train_idx, holdout_idx)
