from __future__ import annotations

import numpy as np

from .core import DEFAULT_SEED, dual_permutation_test


STATES = ("Stable", "Mass-only", "WGAM-only", "Mixed")


def generate_attributions(
    state: str,
    rng: np.random.Generator,
    samples_per_environment: int = 80,
):
    """Generate a controlled attribution experiment with one 3-feature target group.

    The target group has either stable or drifting total mass and either stable or
    migrating internal allocation. Five background features absorb the remaining
    attribution mass.
    """
    if state not in STATES:
        raise ValueError(f"Unknown state: {state}")

    environment = np.repeat(np.arange(4), samples_per_environment)

    stable_mass = np.array([0.30, 0.30, 0.30, 0.30])
    drifting_mass = np.array([0.225, 0.275, 0.325, 0.375])
    stable_q = np.tile(np.array([0.60, 0.25, 0.15]), (4, 1))
    migrating_q = np.array(
        [
            [0.92, 0.05, 0.03],
            [0.05, 0.92, 0.03],
            [0.03, 0.05, 0.92],
            [0.80, 0.04, 0.16],
        ]
    )

    masses = drifting_mass if state in ("Mass-only", "Mixed") else stable_mass
    allocations = migrating_q if state in ("WGAM-only", "Mixed") else stable_q

    rows = []
    for e in range(4):
        for _ in range(samples_per_environment):
            mass = np.clip(rng.normal(masses[e], 0.012), 0.08, 0.75)
            q = rng.dirichlet(allocations[e] * 800.0)
            target = mass * q
            background = (1.0 - mass) * rng.dirichlet(np.full(5, 80.0))
            rows.append(np.concatenate([target, background]))

    return np.vstack(rows), environment, [np.array([0, 1, 2])]


def run_repetition(
    state: str,
    repetition_seed: int,
    permutations: int = 1000,
):
    rng = np.random.default_rng(repetition_seed)
    attributions, environment, groups = generate_attributions(state, rng)
    result = dual_permutation_test(
        attributions,
        environment,
        groups,
        permutations=permutations,
        seed=repetition_seed + 1,
    ).table.iloc[0]
    return {
        "state": state,
        "pred": result["state"],
        "p_mass": result["p_mass"],
        "p_alloc": result["p_alloc"],
        "mass_cv": result["mass_cv"],
        "alloc_jsd": result["alloc_jsd"],
        "mass_ratio": result["mass_effect_ratio"],
        "alloc_ratio": result["alloc_effect_ratio"],
    }


def run_experiment(
    repetitions: int = 100,
    permutations: int = 1000,
    seed: int = DEFAULT_SEED,
):
    rows = []
    for state_index, state in enumerate(STATES):
        for r in range(repetitions):
            repetition_seed = seed + state_index * 10000 + r * 17
            rows.append(run_repetition(state, repetition_seed, permutations=permutations))
    return rows
