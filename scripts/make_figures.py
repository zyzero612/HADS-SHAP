#!/usr/bin/env python3
"""Recreate manuscript Figures 2 and 3 from the reported group results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference" / "beijing"
OUTPUT = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.edgecolor": "#52606B",
        "axes.linewidth": 0.8,
        "xtick.color": "#3F4C56",
        "ytick.color": "#24323D",
        "text.color": "#24323D",
        "svg.fonttype": "none",  # Text stays editable in the SVG.
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def save(fig, stem: str) -> None:
    for ext, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 300},
    }.items():
        fig.savefig(OUTPUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def rank_vs_allocation() -> None:
    data = pd.concat(
        [
            pd.read_csv(REFERENCE / f"{station}_groups.csv").assign(target=station)
            for station in ("Aotizhongxin", "Changping", "Dingling")
        ],
        ignore_index=True,
    )
    data = data.loc[data["state"].eq("WGAM-only")]
    assert len(data) == 51 and data["size"].eq(2).sum() == 29
    assert data["q_alloc"].lt(0.05).all()
    site_colors = {
        "Aotizhongxin": "#365E88",
        "Changping": "#4A938F",
        "Dingling": "#B58B60",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharex=True, sharey=True)
    for ax, mask, title in zip(
        axes,
        (data["size"].eq(2), data["size"].ge(3)),
        ("Two-member groups", "Groups with three or more members"),
    ):
        subset = data[mask]
        for site, color in site_colors.items():
            part = subset[subset["target"].eq(site)]
            ax.scatter(
                part["within_rank_spearman"], part["alloc_jsd"],
                s=37, c=color, alpha=0.83, linewidths=0.4,
                edgecolors="white", label=site,
            )
        ax.set_title(
            title + "\n" + f"n = {len(subset)}; median rank agreement = "
            f"{subset['within_rank_spearman'].median():.3f}",
            fontsize=9.4, pad=10,
        )
        ax.set_xlim(-0.13, 1.045)
        ax.set_ylim(0, 0.205)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.grid(color="#E4EAEB", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Within-group allocation JSD")
    fig.supxlabel("Mean pairwise within-group rank Spearman", y=0.15, fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.53, 0.02), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.30, wspace=0.10)
    save(fig, "fig2_rank_vs_allocation")


ENVIRONMENT_ORDER = [
    "2015_Spring", "2015_Summer", "2015_Autumn", "2016_Winter",
    "2016_Spring", "2016_Summer", "2016_Autumn", "2017_Winter",
]

FEATURES = [
    ("Huairou_PRES_lag12", "Huairou · lag 12 h", "#365E88"),
    ("Wanshouxigong_PRES_lag12", "Wanshouxigong · lag 12 h", "#5B9DA5"),
    ("Huairou_PRES_lag6", "Huairou · lag 6 h", "#829D73"),
    ("Gucheng_PRES_lag6", "Gucheng · lag 6 h", "#BA8B60"),
    ("Wanshouxigong_PRES_lag3", "Wanshouxigong · lag 3 h", "#8A7BA2"),
    ("Dongsi_PRES_lag12", "Dongsi · lag 12 h", "#BF837B"),
    ("Other group members", "Other 42 members", "#D9E0E0"),
]


def pressure_group_case() -> None:
    raw = pd.read_csv(REFERENCE / "aotizhongxin_wgam_case_pressure_group.csv")
    assert set(raw["environment"]) == set(ENVIRONMENT_ORDER)
    data = raw.set_index("environment").loc[ENVIRONMENT_ORDER]
    assert np.allclose(data[[f[0] for f in FEATURES]].sum(axis=1), 1, atol=1e-8)
    assert np.isclose(data["group_mass"].mean(), 0.0978817, atol=1e-5)

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(8.7, 6.15),
        gridspec_kw={"height_ratios": [1, 2.1], "hspace": 0.23},
    )
    x = np.arange(len(data))
    top.plot(x, data["group_mass"], color="#365E88", linewidth=1.7,
             marker="o", markersize=5.2, zorder=3)
    top.set_ylim(0, 0.125)  # Show the baseline; do not visually overstate changes.
    top.set_yticks([0, 0.04, 0.08, 0.12])
    top.set_ylabel("Group share $M_{g,e}$", labelpad=8)
    top.text(0.0, 1.015, "(a) Group attribution share", transform=top.transAxes, va="bottom",
             fontweight="bold", fontsize=11)
    top.text(1.0, 1.015, r"BH-adjusted $q_{mass}=0.0526$",
             transform=top.transAxes, va="bottom", ha="right", fontsize=9)
    top.grid(axis="y", color="#E2E8E8", linewidth=0.7)
    top.set_axisbelow(True)

    bottoms = np.zeros(len(data))
    for col, label, color in FEATURES:
        h = data[col].to_numpy(dtype=float)
        bottom.bar(x, h, bottom=bottoms, width=0.72, label=label,
                   color=color, edgecolor="white", linewidth=0.5)
        bottoms += h
    bottom.set_ylim(0, 1.03)
    bottom.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bottom.set_ylabel("Within-group share $q_{j|g,e}$", labelpad=8)
    bottom.text(0.0, 1.015, "(b) Within-group attribution allocation",
                transform=bottom.transAxes, va="bottom", fontweight="bold", fontsize=11)
    bottom.text(1.0, 1.015, r"BH-adjusted $q_{alloc}=0.0020$",
                transform=bottom.transAxes, va="bottom", ha="right", fontsize=9)
    bottom.set_xticks(x, [s.split("_")[1] + "\n" + s.split("_")[0] for s in ENVIRONMENT_ORDER])
    bottom.set_xlabel("Audit environment (chronological order)")
    bottom.legend(
        ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
        frameon=False, fontsize=8.4, columnspacing=1.7, handlelength=1.8,
    )
    for ax in (top, bottom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.29)
    save(fig, "fig3_pressure_group_case")


if __name__ == "__main__":
    rank_vs_allocation()
    pressure_group_case()
    print(f"Figures saved to {OUTPUT}")
