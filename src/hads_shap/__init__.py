"""HADS-SHAP: hierarchical attribution drift diagnostics for correlated features."""

from .core import (
    bh_fdr,
    cluster_features,
    environment_profiles,
    allocation_active_groups,
    group_statistics,
    dual_permutation_test,
)

__all__ = [
    "bh_fdr",
    "cluster_features",
    "environment_profiles",
    "allocation_active_groups",
    "group_statistics",
    "dual_permutation_test",
]
