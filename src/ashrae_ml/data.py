from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CLASS_LABELS, FEATURE_COLUMNS, PHYSICAL_RANGES


@dataclass
class DataQualityReport:
    initial_rows: int
    after_range_filter_rows: int
    after_target_mapping_rows: int
    dropped_transition_band_rows: int


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_required_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_domain_tr_fallback(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tr" in out.columns and "ta" in out.columns:
        out["tr"] = out["tr"].where(~out["tr"].isna(), out["ta"])
    return out


def apply_physical_range_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, rule in PHYSICAL_RANGES.items():
        if col in out.columns:
            out = out[out[col].isna() | ((out[col] >= rule.minimum) & (out[col] <= rule.maximum))]
    return out


def map_tsv_to_class(tsv: pd.Series) -> pd.Series:
    y = pd.Series(index=tsv.index, dtype="object")
    y.loc[tsv <= -1.0] = CLASS_LABELS[0]
    y.loc[(tsv >= -0.5) & (tsv <= 0.5)] = CLASS_LABELS[1]
    y.loc[tsv >= 1.0] = CLASS_LABELS[2]
    return y


def prepare_classification_frame(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    feature_cols = feature_cols or FEATURE_COLUMNS
    ensure_required_columns(df, feature_cols + [target_col])

    initial_rows = len(df)
    out = apply_domain_tr_fallback(df)
    out = apply_physical_range_filters(out)
    after_range_rows = len(out)

    mapped = map_tsv_to_class(out[target_col])
    dropped_transition = int(mapped.isna().sum())

    out = out.assign(target_class=mapped).dropna(subset=["target_class"])
    out = out.dropna(subset=feature_cols)

    report = DataQualityReport(
        initial_rows=initial_rows,
        after_range_filter_rows=after_range_rows,
        after_target_mapping_rows=len(out),
        dropped_transition_band_rows=dropped_transition,
    )
    return out, report


def build_pmv_baseline(
    df: pd.DataFrame,
    pmv_col: str,
) -> Optional[pd.Series]:
    if pmv_col not in df.columns:
        return None
    return map_tsv_to_class(df[pmv_col])


def save_data_quality_report(path: str, report: DataQualityReport) -> None:
    payload: Dict[str, int] = {
        "initial_rows": report.initial_rows,
        "after_range_filter_rows": report.after_range_filter_rows,
        "after_target_mapping_rows": report.after_target_mapping_rows,
        "dropped_transition_band_rows": report.dropped_transition_band_rows,
    }
    pd.DataFrame([payload]).to_csv(path, index=False)
