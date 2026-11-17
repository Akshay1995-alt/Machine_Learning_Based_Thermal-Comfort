from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FEATURE_COLUMNS
from .data import load_dataset, prepare_classification_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Starter EDA for ASHRAE thermal comfort project.")
    parser.add_argument("--data", required=True, help="Path to raw CSV dataset.")
    parser.add_argument("--target-col", default="tsv", help="TSV column name.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("reports", "figures"),
        help="Directory for output figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.data)
    clean_df, _ = prepare_classification_frame(df, target_col=args.target_col, feature_cols=FEATURE_COLUMNS)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=clean_df, x="target_class", order=["cold", "neutral", "hot"])
    plt.title("Class Balance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "class_balance.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=clean_df.sample(min(8000, len(clean_df)), random_state=42), x="ta", y="rh", hue="target_class", alpha=0.6)
    plt.title("ta vs rh by class")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ta_rh_scatter_by_class.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
