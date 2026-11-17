from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, DEFAULT_TARGET_COL
from .data import load_dataset, prepare_classification_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature importance and SHAP summaries.")
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument("--model-path", required=True, help="Path to trained model .pkl file.")
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="TSV column name.")
    parser.add_argument("--reports-dir", default="reports", help="Output directory for interpretation reports.")
    return parser.parse_args()


def save_feature_importance(model: object, output_path: str) -> bool:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return False

    payload = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    payload.to_csv(output_path, index=False)
    return True


def save_shap_summary(model: object, X_sample: pd.DataFrame, output_path: str) -> bool:
    try:
        import shap  # type: ignore
    except Exception:
        return False

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            arr = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
        else:
            arr = np.mean(np.abs(shap_values), axis=0)

        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": arr}).sort_values(
            "mean_abs_shap", ascending=False
        ).to_csv(output_path, index=False)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)

    with open(args.model_path, "rb") as f:
        model = pickle.load(f)

    raw_df = load_dataset(args.data)
    clean_df, _ = prepare_classification_frame(raw_df, target_col=args.target_col, feature_cols=FEATURE_COLUMNS)
    X = clean_df[FEATURE_COLUMNS].sample(min(len(clean_df), 3000), random_state=42)

    save_feature_importance(
        model=model,
        output_path=os.path.join(args.reports_dir, "feature_importance.csv"),
    )
    save_shap_summary(
        model=model,
        X_sample=X,
        output_path=os.path.join(args.reports_dir, "shap_summary.csv"),
    )
    print("Interpretability exports complete.")


if __name__ == "__main__":
    main()
