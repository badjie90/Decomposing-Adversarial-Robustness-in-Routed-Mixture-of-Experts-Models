#!/usr/bin/env python3
"""Generate the three compact figures used in the revised manuscript.

The script reads only the audited aggregate tables written by the revised
evaluation pipeline.  It deliberately selects the direct-attack records with
no surrogate acquisition count.  The transfer-PGD sensitivity curve uses the
fixed 2,000-label surrogate so that each epsilon uses the same surrogate
setting.

Run from any directory with:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

# The server home directory does not provide a writable Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MANUSCRIPT_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = MANUSCRIPT_DIR / "reference_results" / "tables"
OUTPUT_DIR = MANUSCRIPT_DIR / "results" / "paper" / "Figures"

EPSILON_PRIMARY = 8 / 255
EPSILON_LEVELS = (2 / 255, 4 / 255, 8 / 255, 0.09)
DATASETS = ("bdd100k", "nuimages")
ATTACKS_DIRECT = ("square", "hsj")
DISPLAY_DATASET = {"bdd100k": "BDD100K", "nuimages": "nuImages"}
DISPLAY_ATTACK = {
    "square": "Square",
    "hsj": "HSJ",
    "transfer_pgd": "Transfer-PGD",
}

# A restrained blue scale keeps the new figures consistent with the revision
# marking while remaining legible in grayscale print.
BLUE_DARK = "#083B66"
BLUE_MEDIUM = "#2166AC"
BLUE_LIGHT = "#5A9BD4"
BLUE_PALE = "#A7C9E8"
BLUE_VERY_PALE = "#E6F0F8"
CONDITION_COLORS = (BLUE_DARK, "#2C7FB8", "#73A9D8", "#B5D4EA")
ATTACK_COLORS = {
    "square": BLUE_DARK,
    "hsj": BLUE_MEDIUM,
    "transfer_pgd": BLUE_LIGHT,
}
ATTACK_MARKERS = {"square": "o", "hsj": "s", "transfer_pgd": "^"}
ATTACK_LINESTYLES = {"square": "-", "hsj": "--", "transfer_pgd": ":"}


def configure_style() -> None:
    """Use fonts and line weights that remain clear at journal column width."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 160,
            "savefig.dpi": 360,
        }
    )


def close_to(series: pd.Series, value: float) -> pd.Series:
    """Return a tolerant equality mask for decimal values written to CSV."""
    return np.isclose(series.astype(float), value, rtol=0.0, atol=1e-10)


def read_summary() -> pd.DataFrame:
    path = TABLES_DIR / "metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results table: {path}")
    return pd.read_csv(path)


def read_query_budget() -> pd.DataFrame:
    path = TABLES_DIR / "query_budget_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results table: {path}")
    return pd.read_csv(path)


def direct_base(df: pd.DataFrame, *, epsilon: float) -> pd.DataFrame:
    """Select CSA direct router attacks at one epsilon.

    Direct attacks have no surrogate-acquisition setting.  Keeping that
    condition explicit protects the figures against accidental inclusion of
    transfer-PGD scaling records.
    """
    return df.loc[
        (df["model"] == "csa")
        & (df["architecture"] == "csa_gmoe")
        & (df["target"] == "router")
        & df["dataset"].isin(DATASETS)
        & df["attack"].isin(ATTACKS_DIRECT)
        & close_to(df["epsilon"], epsilon)
        & df["surrogate_acquisition_n"].isna()
    ].copy()


def require_one(df: pd.DataFrame, **criteria: object) -> pd.Series:
    """Fetch exactly one aggregate result and fail loudly on ambiguity."""
    mask = pd.Series(True, index=df.index)
    for field, value in criteria.items():
        if field == "epsilon":
            mask &= close_to(df[field], float(value))
        elif value is None:
            mask &= df[field].isna()
        else:
            mask &= df[field] == value
    result = df.loc[mask]
    if len(result) != 1:
        description = ", ".join(f"{key}={value!r}" for key, value in criteria.items())
        raise ValueError(f"Expected one result for {description}; found {len(result)}")
    return result.iloc[0]


def finalize_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Apply a minimal journal-style axis treatment."""
    ax.grid(axis=grid_axis, color="#D7E3EE", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.0, width=0.6)


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save a vector PDF and a high-resolution PNG with the same basename."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return pdf_path, png_path


def plot_controlled_attribution(summary: pd.DataFrame) -> tuple[Path, Path]:
    """Plot CC/CA/AC/AA macro-F1 with paired-bootstrap confidence intervals."""
    data = direct_base(summary, epsilon=EPSILON_PRIMARY)
    conditions = ("CC", "CA", "AC", "AA")
    condition_labels = ("CC", "CA", "AC", "AA")
    panels = (("square", "bdd100k"), ("square", "nuimages"), ("hsj", "bdd100k"), ("hsj", "nuimages"))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.75), sharey=True)
    x = np.arange(len(conditions))
    for index, (attack, dataset) in enumerate(panels):
        ax = axes.flat[index]
        rows = [
            require_one(
                data,
                dataset=dataset,
                attack=attack,
                metric=f"{condition}/macro_f1",
            )
            for condition in conditions
        ]
        means = np.array([row["mean"] for row in rows], dtype=float)
        low = np.array([row["ci_low"] for row in rows], dtype=float)
        high = np.array([row["ci_high"] for row in rows], dtype=float)
        errors = np.vstack((means - low, high - means))

        for position, mean, lo, hi, shade in zip(x, means, low, high, CONDITION_COLORS):
            ax.errorbar(
                position, mean, yerr=[[mean - lo], [hi - mean]],
                fmt="o", color=BLUE_DARK, markerfacecolor=shade,
                markeredgecolor=BLUE_DARK, markersize=6,
                capsize=3.5, capthick=1.1, linewidth=1.1, zorder=3,
            )
        ax.set_title(f"{DISPLAY_DATASET[dataset]}, {DISPLAY_ATTACK[attack]}", pad=4)
        ax.set_xticks(x, condition_labels)
        ax.set_ylim(0.48, 0.87)
        ax.set_yticks((0.5, 0.6, 0.7, 0.8))
        if index % 2 == 0:
            ax.set_ylabel("Macro-F1")
        finalize_axis(ax)

    fig.text(
        0.5,
        0.012,
        "State order is expert features, routing. C denotes clean and A denotes adversarial.",
        ha="center",
        va="bottom",
        color=BLUE_DARK,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.115, hspace=0.36, wspace=0.16)
    return save_figure(fig, "controlled_attribution")


def sensitivity_rows(summary: pd.DataFrame, dataset: str, attack: str, metric: str) -> list[pd.Series]:
    """Select one sensitivity result at each epsilon using a fixed attack setup."""
    rows: list[pd.Series] = []
    for epsilon in EPSILON_LEVELS:
        criteria: dict[str, object] = {
            "dataset": dataset,
            "model": "csa",
            "architecture": "csa_gmoe",
            "target": "router",
            "attack": attack,
            "epsilon": epsilon,
            "metric": metric,
        }
        # The sensitivity configuration fixes Transfer-PGD to the 2,000-label
        # surrogate. At epsilon=8/255, the table also carries scaling records,
        # so this condition is needed to avoid mixing experiments.
        criteria["surrogate_acquisition_n"] = 2000.0 if attack == "transfer_pgd" else None
        rows.append(require_one(summary, **criteria))
    return rows


def plot_epsilon_sensitivity(summary: pd.DataFrame) -> tuple[Path, Path]:
    """Plot ASR and final AA macro-F1 over the four perturbation budgets."""
    metrics = (
        ("routing/router_asr", "Router ASR"),
        ("AA/macro_f1", "AA macro-F1"),
    )
    attacks = ("square", "hsj", "transfer_pgd")
    x = np.arange(len(EPSILON_LEVELS))
    xlabels = (r"$2/255$", r"$4/255$", r"$8/255$", "$0.09$\n($\\approx23/255$)")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), sharex=True, sharey="row")
    handles: list[plt.Line2D] = []
    for col, dataset in enumerate(DATASETS):
        for row_index, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_index, col]
            for attack in attacks:
                rows = sensitivity_rows(summary, dataset, attack, metric)
                means = np.array([entry["mean"] for entry in rows], dtype=float)
                low = np.array([entry["ci_low"] for entry in rows], dtype=float)
                high = np.array([entry["ci_high"] for entry in rows], dtype=float)
                (line,) = ax.plot(
                    x,
                    means,
                    color=ATTACK_COLORS[attack],
                    marker=ATTACK_MARKERS[attack],
                    linestyle=ATTACK_LINESTYLES[attack],
                    label=DISPLAY_ATTACK[attack],
                    zorder=3,
                )
                ax.fill_between(x, low, high, color=ATTACK_COLORS[attack], alpha=0.13, linewidth=0, zorder=1)
                if row_index == 0 and col == 0:
                    handles.append(line)
            if row_index == 0:
                ax.set_title(DISPLAY_DATASET[dataset], pad=4)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(-0.12, len(EPSILON_LEVELS) - 0.88)
            ax.set_ylim(0.0, 1.03)
            ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
            ax.set_xticks(x, xlabels if row_index == 1 else ())
            if row_index == 1:
                ax.set_xlabel(r"Perturbation budget $\epsilon$")
            finalize_axis(ax)

    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.012),
        frameon=False,
        handlelength=2.3,
        columnspacing=1.7,
    )
    fig.text(
        0.5,
        0.06,
        "Transfer-PGD uses the fixed 2,000-label surrogate. Shading denotes 95% paired-bootstrap intervals.",
        ha="center",
        va="bottom",
        color=BLUE_DARK,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.20, hspace=0.27, wspace=0.16)
    return save_figure(fig, "epsilon_sensitivity")


def plot_query_budget(query: pd.DataFrame) -> tuple[Path, Path]:
    """Plot router ASR versus requested query budget for direct CSA attacks."""
    data = direct_base(query, epsilon=EPSILON_PRIMARY)
    data = data.loc[data["metric"] == "routing/router_asr"].copy()
    budgets = np.array((100, 250, 500, 1000), dtype=float)
    panels = (("square", "bdd100k"), ("square", "nuimages"), ("hsj", "bdd100k"), ("hsj", "nuimages"))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.65), sharex=True, sharey=True)
    for index, (attack, dataset) in enumerate(panels):
        ax = axes.flat[index]
        rows = [
            require_one(
                data,
                dataset=dataset,
                attack=attack,
                requested_query_budget=int(budget),
            )
            for budget in budgets
        ]
        means = np.array([row["mean"] for row in rows], dtype=float)
        low = np.array([row["ci_low"] for row in rows], dtype=float)
        high = np.array([row["ci_high"] for row in rows], dtype=float)
        ax.plot(budgets, means, color=BLUE_DARK, marker="o", zorder=3)
        ax.fill_between(budgets, low, high, color=BLUE_PALE, alpha=0.72, linewidth=0, zorder=1)
        ax.set_title(f"{DISPLAY_DATASET[dataset]}, {DISPLAY_ATTACK[attack]}", pad=4)
        ax.set_xlim(75, 1025)
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks(budgets)
        ax.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        if index % 2 == 0:
            ax.set_ylabel("Router ASR")
        if index >= 2:
            ax.set_xlabel("Requested query budget")
        finalize_axis(ax)

    fig.text(
        0.5,
        0.012,
        r"CSA router-targeted attacks at $\epsilon=8/255$. Shading denotes 95% paired-bootstrap intervals.",
        ha="center",
        va="bottom",
        color=BLUE_DARK,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.12, hspace=0.37, wspace=0.16)
    return save_figure(fig, "query_budget")


def main() -> None:
    global TABLES_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    TABLES_DIR, OUTPUT_DIR = args.tables_dir.resolve(), args.output_dir.resolve()
    configure_style()
    summary = read_summary()
    query = read_query_budget()
    outputs = (
        plot_controlled_attribution(summary),
        plot_epsilon_sensitivity(summary),
        plot_query_budget(query),
    )
    for pdf_path, png_path in outputs:
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
