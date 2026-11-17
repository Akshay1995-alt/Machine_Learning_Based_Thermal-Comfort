from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, float | str]:
    return {
        "model": model_name,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str],
    output_csv: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_csv)


def save_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: List[str],
    output_csv: str,
) -> None:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_csv)
