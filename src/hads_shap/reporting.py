from __future__ import annotations

import numpy as np
import pandas as pd


def state_summary(table: pd.DataFrame) -> dict:
    counts = table["state"].value_counts()
    n = len(table)
    return {
        "n_active_groups": n,
        "Stable": int(counts.get("Stable", 0)),
        "Mass-only": int(counts.get("Mass-only", 0)),
        "WGAM-only": int(counts.get("WGAM-only", 0)),
        "Mixed": int(counts.get("Mixed", 0)),
        "any_wgam_fraction": float(table.state.isin(["WGAM-only", "Mixed"]).mean()),
        "wgam_only_fraction": float((table.state == "WGAM-only").mean()),
    }
