from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .core import DEFAULT_SEED, cluster_features, dual_permutation_test


VARIABLES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
LAGS = [1, 3, 6, 12, 24, 48]
STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng",
    "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]
TARGETS = ["Aotizhongxin", "Changping", "Dingling"]
ENVIRONMENTS = [
    (2015, "Spring"), (2015, "Summer"), (2015, "Autumn"),
    (2016, "Winter"), (2016, "Spring"), (2016, "Summer"),
    (2016, "Autumn"), (2017, "Winter"),
]


def load_wide(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
    frames = []
    with zipfile.ZipFile(zip_path) as archive:
        for station in STATIONS:
            name = (
                "PRSA_Data_20130301-20170228/"
                f"PRSA_Data_{station}_20130301-20170228.csv"
            )
            df = pd.read_csv(archive.open(name))
            dt = pd.to_datetime(
                dict(year=df.year, month=df.month, day=df.day, hour=df.hour)
            )
            values = df[VARIABLES].copy()
            values.columns = [f"{station}_{v}" for v in VARIABLES]
            values.index = dt
            frames.append(values)
    return pd.concat(frames, axis=1).sort_index()


def make_features(
    wide: pd.DataFrame,
    target_station: str,
    horizon: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    lagged = []
    for lag in LAGS:
        frame = wide.shift(lag).copy()
        frame.columns = [f"{c}_lag{lag}" for c in frame.columns]
        lagged.append(frame)
    X = pd.concat(lagged, axis=1)
    y = wide[f"{target_station}_PM2.5"].shift(-horizon).rename("target")
    return X, y


def _season_year(timestamp: pd.Timestamp) -> tuple[int, str]:
    month = timestamp.month
    if month in (12, 1, 2):
        return (timestamp.year + 1 if month == 12 else timestamp.year, "Winter")
    if month in (3, 4, 5):
        return timestamp.year, "Spring"
    if month in (6, 7, 8):
        return timestamp.year, "Summer"
    return timestamp.year, "Autumn"


def weekly_audit_blocks(index: pd.DatetimeIndex, weeks_per_environment: int = 12):
    block_id = np.full(len(index), -1, dtype=int)
    environment = np.array([""] * len(index), dtype=object)
    labels = np.array([_season_year(t) for t in index], dtype=object)

    current_block = 0
    for year, season in ENVIRONMENTS:
        mask = np.array([(a == year and b == season) for a, b in labels])
        positions = np.flatnonzero(mask)
        times = index[positions]
        starts = [t for t in times if t.weekday() == 0 and t.hour == 0]
        if not starts:
            raise RuntimeError(f"No complete weekly block start for {year} {season}")

        start = starts[0]
        collected = 0
        offset = 0
        while collected < weeks_per_environment and offset < 20:
            left = start + pd.Timedelta(hours=168 * offset)
            right = left + pd.Timedelta(hours=168)
            offset += 1
            block_positions = np.flatnonzero((index >= left) & (index < right) & mask)
            if len(block_positions) == 168:
                block_id[block_positions] = current_block
                environment[block_positions] = f"{year}_{season}"
                current_block += 1
                collected += 1

        if collected < weeks_per_environment:
            raise RuntimeError(f"Only {collected} complete weeks for {year} {season}")

    keep = block_id >= 0
    expected = len(ENVIRONMENTS) * weeks_per_environment
    if len(np.unique(block_id[keep])) != expected:
        raise RuntimeError(f"Expected {expected} weekly blocks")
    return keep, block_id, environment


def prepare_split(
    wide: pd.DataFrame,
    target_station: str,
    horizon: int = 24,
):
    X, y = make_features(wide, target_station, horizon=horizon)
    train_mask = (X.index < "2015-01-01") & (X.index >= "2013-03-05")
    eval_mask = (X.index >= "2015-03-01") & (X.index < "2017-03-01")

    medians = X.loc[train_mask].median(axis=0)
    X_train = X.loc[train_mask].fillna(medians).astype(np.float32)
    y_train = y.loc[train_mask]
    valid_train = y_train.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]

    X_eval = X.loc[eval_mask].fillna(medians).astype(np.float32)
    y_eval = y.loc[eval_mask]
    return X_train, y_train, X_eval, y_eval


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, seed: int = DEFAULT_SEED):
    model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=8,
    )
    model.fit(X_train, y_train)
    return model


def training_groups(
    X_train: pd.DataFrame,
    threshold: float = 0.90,
    seed: int = DEFAULT_SEED,
    sample_size: int = 5000,
):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
    sampled = X_train.iloc[indices]
    return cluster_features(sampled, threshold=threshold)


def attribution_blocks(model, X_eval: pd.DataFrame):
    keep, block_ids, environments = weekly_audit_blocks(X_eval.index)
    selected = np.zeros(len(X_eval), dtype=bool)
    for block in np.unique(block_ids[keep]):
        positions = np.flatnonzero(block_ids == block)[::3]
        selected[positions] = True

    X_audit = X_eval.iloc[np.flatnonzero(selected)]
    contributions = model.get_booster().predict(
        xgb.DMatrix(X_audit), pred_contribs=True
    )
    absolute = np.abs(contributions[:, :-1])

    selected_blocks = block_ids[selected]
    selected_env = environments[selected]
    block_attributions = []
    block_environments = []
    for block in np.unique(selected_blocks):
        mask = selected_blocks == block
        block_attributions.append(absolute[mask].mean(axis=0))
        block_environments.append(selected_env[mask][0])

    return np.vstack(block_attributions), np.asarray(block_environments)


def run_xgboost_target(
    zip_path: str | Path,
    target_station: str,
    horizon: int = 24,
    threshold: float = 0.90,
    permutations: int = 1000,
    seed: int = DEFAULT_SEED,
):
    wide = load_wide(zip_path)
    X_train, y_train, X_eval, y_eval = prepare_split(
        wide, target_station, horizon=horizon
    )
    model = fit_xgboost(X_train, y_train, seed=seed)

    prediction = model.predict(X_eval)
    valid = y_eval.notna().to_numpy()
    performance = {
        "station": target_station,
        "horizon_h": horizon,
        "n_train": len(X_train),
        "n_eval": int(valid.sum()),
        "rmse": float(np.sqrt(mean_squared_error(y_eval.to_numpy()[valid], prediction[valid]))),
        "mae": float(mean_absolute_error(y_eval.to_numpy()[valid], prediction[valid])),
        "r2": float(r2_score(y_eval.to_numpy()[valid], prediction[valid])),
    }

    groups, _ = training_groups(X_train, threshold=threshold, seed=seed)
    block_A, block_env = attribution_blocks(model, X_eval)
    result = dual_permutation_test(
        block_A,
        block_env,
        groups,
        permutations=permutations,
        seed=seed,
    )

    names = np.asarray(X_train.columns)
    result.table["member_names"] = result.table.members.map(
        lambda s: ";".join(names[[int(v) for v in s.split(";")]])
    )
    return performance, result
