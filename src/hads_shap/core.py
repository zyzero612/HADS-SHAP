from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from numba import njit


DEFAULT_SEED = 20260926


@dataclass(frozen=True)
class PermutationResult:
    table: pd.DataFrame
    environment_profiles: np.ndarray
    excluded_groups: int


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    previous = 1.0
    for i in range(n - 1, -1, -1):
        previous = min(previous, ranked[i] * n / (i + 1))
        q[order[i]] = min(previous, 1.0)
    return q


def cluster_features(
    X: pd.DataFrame | np.ndarray,
    threshold: float = 0.90,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Complete-linkage clustering from absolute Spearman correlation."""
    frame = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
    corr = frame.corr(method="spearman").to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return groups, corr


def _encode(values: Sequence):
    unique, codes = np.unique(np.asarray(values), return_inverse=True)
    return unique, codes


def _profiles_from_codes(
    A: np.ndarray,
    env_codes: np.ndarray,
    n_env: int,
    strata_codes: np.ndarray | None = None,
    n_strata: int | None = None,
) -> np.ndarray:
    """Fast normalized environment profiles from integer labels."""
    n_features = A.shape[1]
    if strata_codes is None:
        sums = np.zeros((n_env, n_features), dtype=float)
        np.add.at(sums, env_codes, A)
        counts = np.bincount(env_codes, minlength=n_env).astype(float)
        means = sums / counts[:, None]
    else:
        if n_strata is None:
            raise ValueError("n_strata is required when strata_codes are supplied")
        combined = env_codes * n_strata + strata_codes
        n_cells = n_env * n_strata
        sums = np.zeros((n_cells, n_features), dtype=float)
        np.add.at(sums, combined, A)
        counts = np.bincount(combined, minlength=n_cells).astype(float)
        means = np.zeros_like(sums)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero, None]
        means = means.reshape(n_env, n_strata, n_features)
        valid = (counts.reshape(n_env, n_strata) > 0)
        # Equal stratum weighting within each environment.
        weighted = np.where(valid[:, :, None], means, 0.0)
        means = weighted.sum(axis=1) / valid.sum(axis=1)[:, None]

    totals = means.sum(axis=1)
    return np.divide(means, totals[:, None], out=np.zeros_like(means), where=totals[:, None] > 0)


def environment_profiles(
    attributions: np.ndarray,
    environments: Sequence,
    environment_values: Sequence | None = None,
    strata: Sequence | None = None,
) -> np.ndarray:
    """Compute normalized mean absolute attribution profiles by environment."""
    A = np.asarray(attributions, dtype=float)
    env = np.asarray(environments)

    if environment_values is None:
        _, env_codes = _encode(env)
        n_env = int(env_codes.max()) + 1
    else:
        environment_values = np.asarray(environment_values)
        lookup = {value: i for i, value in enumerate(environment_values)}
        env_codes = np.array([lookup[value] for value in env], dtype=int)
        n_env = len(environment_values)

    if strata is None:
        return _profiles_from_codes(A, env_codes, n_env)

    _, strata_codes = _encode(strata)
    return _profiles_from_codes(
        A,
        env_codes,
        n_env,
        strata_codes=strata_codes,
        n_strata=int(strata_codes.max()) + 1,
    )

def allocation_active_groups(
    profiles: np.ndarray,
    groups: Iterable[np.ndarray],
    epsilon: float = 1e-12,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split groups into allocation-active and ineligible groups.

    Within-group allocation is defined only when group attribution mass is
    positive in every audited environment.
    """
    active: list[np.ndarray] = []
    excluded: list[np.ndarray] = []
    for group in groups:
        mass = profiles[:, group].sum(axis=1)
        (active if np.all(mass > epsilon) else excluded).append(group)
    return active, excluded


def _allocation_jsd(q: np.ndarray) -> float:
    q_bar = q.mean(axis=0)
    positive_bar = q_bar > 0
    h_bar = -np.sum(q_bar[positive_bar] * np.log2(q_bar[positive_bar]))
    h_rows = np.zeros(q.shape[0], dtype=float)
    for i, row in enumerate(q):
        positive = row > 0
        h_rows[i] = -np.sum(row[positive] * np.log2(row[positive]))
    return float(h_bar - h_rows.mean())

def group_statistics(profiles: np.ndarray, groups: Sequence[np.ndarray]) -> pd.DataFrame:
    """Calculate group-mass and within-group allocation statistics."""
    rows = []
    n_env = profiles.shape[0]

    for group_id, group in enumerate(groups):
        if len(group) < 2:
            continue

        group_profile = profiles[:, group]
        mass = group_profile.sum(axis=1)
        mass_mean = mass.mean()
        mass_cv = mass.std(ddof=1) / (mass_mean + 1e-15)
        q = group_profile / mass[:, None]
        alloc_jsd = _allocation_jsd(q)

        representatives = np.argmax(q, axis=1)
        rank_correlations = []
        for i in range(n_env):
            for j in range(i + 1, n_env):
                rho = spearmanr(q[i], q[j]).statistic
                if np.isfinite(rho):
                    rank_correlations.append(rho)

        rows.append(
            {
                "group_id": group_id,
                "size": len(group),
                "mass_mean": mass_mean,
                "mass_cv": mass_cv,
                "alloc_jsd": alloc_jsd,
                "n_representatives": len(np.unique(representatives)),
                "within_rank_spearman": (
                    float(np.mean(rank_correlations)) if rank_correlations else np.nan
                ),
                "members": ";".join(map(str, group.tolist())),
            }
        )

    return pd.DataFrame(rows)



@njit(cache=True)
def _null_stats_numba(A, permutation_codes, strata_codes, n_env, n_strata, group_label, n_groups):
    B, n_samples = permutation_codes.shape
    n_features = A.shape[1]
    null_mass = np.zeros((B, n_groups), dtype=np.float64)
    null_alloc = np.zeros((B, n_groups), dtype=np.float64)

    use_strata = n_strata > 0
    cells = n_env * (n_strata if use_strata else 1)

    for b in range(B):
        sums = np.zeros((cells, n_features), dtype=np.float64)
        counts = np.zeros(cells, dtype=np.int64)

        for i in range(n_samples):
            e = permutation_codes[b, i]
            if use_strata:
                cell = e * n_strata + strata_codes[i]
            else:
                cell = e
            counts[cell] += 1
            for f in range(n_features):
                sums[cell, f] += A[i, f]

        P = np.zeros((n_env, n_features), dtype=np.float64)
        for e in range(n_env):
            for f in range(n_features):
                value = 0.0
                if use_strata:
                    for c in range(n_strata):
                        cell = e * n_strata + c
                        value += sums[cell, f] / counts[cell]
                    value /= n_strata
                else:
                    value = sums[e, f] / counts[e]
                P[e, f] = value

            total = 0.0
            for f in range(n_features):
                total += P[e, f]
            if total > 0.0:
                for f in range(n_features):
                    P[e, f] /= total

        mass = np.zeros((n_env, n_groups), dtype=np.float64)
        for e in range(n_env):
            for f in range(n_features):
                g = group_label[f]
                if g >= 0:
                    mass[e, g] += P[e, f]

        for g in range(n_groups):
            mean_mass = 0.0
            for e in range(n_env):
                mean_mass += mass[e, g]
            mean_mass /= n_env

            variance = 0.0
            for e in range(n_env):
                d = mass[e, g] - mean_mass
                variance += d * d
            if n_env > 1:
                variance /= (n_env - 1)
            null_mass[b, g] = np.sqrt(variance) / (mean_mass + 1e-15)

        qbar = np.zeros(n_features, dtype=np.float64)
        h_rows = np.zeros((n_env, n_groups), dtype=np.float64)
        inv_log2 = 1.0 / np.log(2.0)

        for e in range(n_env):
            for f in range(n_features):
                g = group_label[f]
                if g >= 0:
                    q = P[e, f] / mass[e, g]
                    qbar[f] += q / n_env
                    if q > 0.0:
                        h_rows[e, g] -= q * np.log(q) * inv_log2

        h_bar = np.zeros(n_groups, dtype=np.float64)
        for f in range(n_features):
            g = group_label[f]
            if g >= 0:
                q = qbar[f]
                if q > 0.0:
                    h_bar[g] -= q * np.log(q) * inv_log2

        for g in range(n_groups):
            mean_h = 0.0
            for e in range(n_env):
                mean_h += h_rows[e, g]
            mean_h /= n_env
            null_alloc[b, g] = h_bar[g] - mean_h

    return null_mass, null_alloc


def _permutation_matrix(env_codes, permutations, rng, strata_codes=None, n_strata=0):
    env_codes = np.asarray(env_codes, dtype=np.int64)
    out = np.empty((permutations, len(env_codes)), dtype=np.int64)
    if strata_codes is None:
        for b in range(permutations):
            out[b] = rng.permutation(env_codes)
    else:
        strata_codes = np.asarray(strata_codes, dtype=np.int64)
        for b in range(permutations):
            row = env_codes.copy()
            for c in range(n_strata):
                idx = np.flatnonzero(strata_codes == c)
                row[idx] = rng.permutation(row[idx])
            out[b] = row
    return out

def dual_permutation_test(
    attributions: np.ndarray,
    environments: Sequence,
    groups: Sequence[np.ndarray],
    permutations: int = 1000,
    strata: Sequence | None = None,
    seed: int = DEFAULT_SEED,
    epsilon: float = 1e-12,
) -> PermutationResult:
    """Separate group-mass drift and within-group allocation drift.

    Environment labels are permuted while attributions and model outputs remain
    fixed. If ``strata`` is supplied, labels are permuted independently inside
    each stratum.
    """
    rng = np.random.default_rng(seed)
    env = np.asarray(environments)
    A = np.asarray(attributions, dtype=float)
    environment_values, env_codes = _encode(env)
    n_env = len(environment_values)
    if strata is None:
        strata_arr = None
        strata_codes = None
        n_strata = None
    else:
        strata_arr = np.asarray(strata)
        _, strata_codes = _encode(strata_arr)
        n_strata = int(strata_codes.max()) + 1

    observed_profiles = _profiles_from_codes(
        A, env_codes, n_env, strata_codes=strata_codes, n_strata=n_strata
    )
    active_groups, excluded_groups = allocation_active_groups(
        observed_profiles, groups, epsilon=epsilon
    )
    tested_groups = [group for group in active_groups if len(group) > 1]
    observed = group_statistics(observed_profiles, tested_groups)

    observed_mass = observed["mass_cv"].to_numpy()
    observed_alloc = observed["alloc_jsd"].to_numpy()
    group_label = np.full(A.shape[1], -1, dtype=np.int64)
    for g, group in enumerate(tested_groups):
        group_label[np.asarray(group, dtype=int)] = g

    permutation_codes = _permutation_matrix(
        env_codes,
        permutations,
        rng,
        strata_codes=strata_codes,
        n_strata=(0 if n_strata is None else n_strata),
    )
    null_mass, null_alloc = _null_stats_numba(
        A,
        permutation_codes,
        np.empty(0, dtype=np.int64) if strata_codes is None else strata_codes.astype(np.int64),
        n_env,
        0 if n_strata is None else n_strata,
        group_label,
        len(tested_groups),
    )

    p_mass = (1 + (null_mass >= observed_mass).sum(axis=0)) / (permutations + 1)
    p_alloc = (1 + (null_alloc >= observed_alloc).sum(axis=0)) / (permutations + 1)

    observed["p_mass"] = p_mass
    observed["p_alloc"] = p_alloc
    observed["q_mass"] = bh_fdr(p_mass)
    observed["q_alloc"] = bh_fdr(p_alloc)
    observed["state"] = np.where(
        (observed.q_mass < 0.05) & (observed.q_alloc < 0.05),
        "Mixed",
        np.where(
            observed.q_mass < 0.05,
            "Mass-only",
            np.where(observed.q_alloc < 0.05, "WGAM-only", "Stable"),
        ),
    )

    observed["mass_null_mean"] = null_mass.mean(axis=0)
    observed["alloc_null_mean"] = null_alloc.mean(axis=0)
    observed["mass_effect_ratio"] = observed.mass_cv / (observed.mass_null_mean + 1e-15)
    observed["alloc_effect_ratio"] = observed.alloc_jsd / (observed.alloc_null_mean + 1e-15)

    return PermutationResult(
        table=observed,
        environment_profiles=observed_profiles,
        excluded_groups=len(excluded_groups),
    )
