from __future__ import annotations

import argparse
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from .config import FEATURE_COLUMNS, DEFAULT_TARGET_COL
from .data import load_dataset, prepare_classification_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Random Forest baseline.")
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="TSV column name.")
    parser.add_argument("--models-dir", default="models", help="Output dir for tuned model.")
    parser.add_argument("--reports-dir", default="reports", help="Output dir for CV results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    raw_df = load_dataset(args.data)
    clean_df, _ = prepare_classification_frame(raw_df, target_col=args.target_col, feature_cols=FEATURE_COLUMNS)
    X = clean_df[FEATURE_COLUMNS]
    y = clean_df["target_class"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    base = RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    params = {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4],
    }

    search = GridSearchCV(
        estimator=base,
        param_grid=params,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        os.path.join(args.reports_dir, "rf_gridsearch_results.csv"),
        index=False,
    )
    with open(os.path.join(args.models_dir, "random_forest_tuned.pkl"), "wb") as f:
        pickle.dump(search.best_estimator_, f)

    print("Best params:", search.best_params_)
    print("Best CV macro F1:", search.best_score_)


if __name__ == "__main__":
    main()
