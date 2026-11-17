from __future__ import annotations

import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CLASS_LABELS, DEFAULT_PMV_COL, DEFAULT_TARGET_COL, FEATURE_COLUMNS
from .data import build_pmv_baseline, load_dataset, prepare_classification_frame, save_data_quality_report
from .evaluate import compute_metrics, save_classification_report, save_confusion_matrix
from .models import get_candidate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phase-1 ASHRAE thermal comfort models.")
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="TSV column name.")
    parser.add_argument("--pmv-col", default=DEFAULT_PMV_COL, help="PMV column name for benchmark.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--models-dir", default="models", help="Directory to save trained models.")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save reports.")
    return parser.parse_args()


def save_metrics_table(path: str, metrics_rows: List[dict]) -> None:
    pd.DataFrame(metrics_rows).sort_values("macro_f1", ascending=False).to_csv(path, index=False)


def run() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    raw_df = load_dataset(args.data)
    clean_df, quality = prepare_classification_frame(raw_df, target_col=args.target_col, feature_cols=FEATURE_COLUMNS)
    save_data_quality_report(os.path.join(args.reports_dir, "data_quality_report.csv"), quality)

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["target_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    labels = list(CLASS_LABELS)
    metrics_rows: List[dict] = []

    models = get_candidate_models(random_state=args.random_state)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_rows.append(compute_metrics(y_test, y_pred, name))
        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            labels=labels,
            output_csv=os.path.join(args.reports_dir, f"confusion_matrix_{name}.csv"),
        )
        save_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            labels=labels,
            output_csv=os.path.join(args.reports_dir, f"classification_report_{name}.csv"),
        )

        model_path = os.path.join(args.models_dir, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    pmv_mapped = build_pmv_baseline(clean_df, args.pmv_col)
    if pmv_mapped is not None:
        benchmark_df = clean_df.copy()
        benchmark_df["pmv_class"] = pmv_mapped
        benchmark_df = benchmark_df.dropna(subset=["pmv_class"])

        if not benchmark_df.empty:
            _, bench_test = train_test_split(
                benchmark_df,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=benchmark_df["target_class"],
            )
            pmv_metrics = compute_metrics(
                y_true=bench_test["target_class"],
                y_pred=bench_test["pmv_class"].to_numpy(),
                model_name="pmv_baseline",
            )
            pd.DataFrame([pmv_metrics]).to_csv(os.path.join(args.reports_dir, "pmv_baseline_metrics.csv"), index=False)
            metrics_rows.append(pmv_metrics)

    save_metrics_table(os.path.join(args.reports_dir, "metrics_summary.csv"), metrics_rows)
    print("Training complete. Reports written to:", args.reports_dir)


if __name__ == "__main__":
    run()
