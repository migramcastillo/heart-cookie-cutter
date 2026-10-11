"""Visualization and EDA functions for the heart disease dataset.

Functions can be imported into notebooks or run directly from the CLI.

Usage from CLI:
    python -m heart_ccds.features
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_ccds.config import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET,
    logger,
)
from heart_ccds.dataset import load_data


def plot_numeric_histograms(
    df,
    features = None,
    save_path = None,
):
    """Plot histograms for numerical features."""
    cols = features if features is not None else NUMERIC_FEATURES
    fig = plt.figure(figsize=(15, 10))
    df[cols].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        logger.info(f"Saved numeric histograms to {out}")

    return fig


def plot_numeric_boxplots(
    df,
    features = None,
    save_path = None,
):
    """Plot boxplots for numerical features to check for outliers."""
    cols = features if features is not None else NUMERIC_FEATURES
    fig = plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        logger.info(f"Saved numeric boxplots to {out}")

    return fig


def plot_categorical_counts(
    df,
    features = None,
    target = TARGET,
    save_path = None,
):
    """Plot count plots for categorical features and target."""
    cat_cols = features if features is not None else CATEGORICAL_FEATURES
    plot_cols = cat_cols + [target] if target in df.columns else cat_cols

    fig = plt.figure(figsize=(15, 12))
    for i, col in enumerate(plot_cols, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=df[col])
        plt.title(f"Countplot of {col}")
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
        logger.info(f"Saved categorical countplots to {out}")

    return fig


def generate_all_eda_plots(df, output_dir = FIGURES_DIR):
    """Generate and save all EDA plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_numeric_histograms(df, save_path=out_dir / "numeric_histograms.png")
    plot_numeric_boxplots(df, save_path=out_dir / "numeric_boxplots.png")
    plot_categorical_counts(df, save_path=out_dir / "categorical_counts.png")
    logger.info(f"Generated all EDA figures in {out_dir}")


def main():
    """CLI entrypoint to generate and save EDA plots."""
    parser = argparse.ArgumentParser(description="Generate EDA visualizations for heart dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_PATH),
        help=f"Path to input CSV data (default: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FIGURES_DIR),
        help=f"Directory to save output figures (default: {FIGURES_DIR})",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    generate_all_eda_plots(df, args.output_dir)


if __name__ == "__main__":
    main()
