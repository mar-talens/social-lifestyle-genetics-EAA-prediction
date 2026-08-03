"""Supplementary Figure S1: correlations among epigenetic age measures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("output/epigenetic_age_events.csv")
OUTPUT_FILE = Path("output/figures/SuppFig1_clock_correlations.png")


def plot_corr(ax, data_frame, columns, labels):
    """Plot one lower-triangle Pearson correlation heatmap."""
    present = [column for column in columns if column in data_frame.columns]
    complete_cases = data_frame[present].dropna()
    correlation = complete_cases.corr().rename(index=labels, columns=labels)

    # The notebook hides the strictly upper triangle and leaves the diagonal visible.
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)

    sns.heatmap(
        correlation,
        annot=True,
        cmap="Reds",
        fmt=".2f",
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar=False,
        ax=ax,
    )


def main():
    df = pd.read_csv(INPUT_FILE)

    # Panel A: chronological age and DNAm age measures.
    age_columns = [
        "CHRON_AGE",
        "LEVINE_DNAMAGE",
        "DNAMGRIMAGE",
        "MPOA",
        "HORVATH_DNAMAGE",
        "HANNUM_DNAMAGE",
    ]
    age_labels = {
        "CHRON_AGE": "Age",
        "LEVINE_DNAMAGE": "PhenoAge",
        "DNAMGRIMAGE": "GrimAge",
        "MPOA": "DunedinPoAm",
        "HORVATH_DNAMAGE": "Horvath",
        "HANNUM_DNAMAGE": "Hannum",
    }

    # Panel B: epigenetic age acceleration measures.
    eaa_columns = [
        "EAA_LEVINE",
        "EAA_GRIMAGE",
        "EAA_DUNEDINMPOA",
        "EAA_HORVATH",
        "EAA_HANNUM",
    ]
    eaa_labels = {
        "EAA_LEVINE": "PhenoAge",
        "EAA_GRIMAGE": "GrimAge",
        "EAA_DUNEDINMPOA": "DunedinPoAm",
        "EAA_HORVATH": "Horvath",
        "EAA_HANNUM": "Hannum",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_corr(axes[0], df, age_columns, age_labels)
    plot_corr(axes[1], df, eaa_columns, eaa_labels)

    axes[0].text(
        -0.15,
        1.05,
        "A",
        transform=axes[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )
    axes[1].text(
        -0.15,
        1.05,
        "B",
        transform=axes[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # The notebook displayed the figure without export settings; 300 dpi is
    # newly specified here as a reproducible repository export convention.
    fig.savefig(OUTPUT_FILE, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
