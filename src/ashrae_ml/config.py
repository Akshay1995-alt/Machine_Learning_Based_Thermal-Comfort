from dataclasses import dataclass
from typing import Dict, List, Tuple


FEATURE_COLUMNS: List[str] = ["ta", "rh", "vel", "tr", "clo", "met"]
DEFAULT_TARGET_COL = "tsv"
DEFAULT_PMV_COL = "pmv"


@dataclass(frozen=True)
class RangeRule:
    minimum: float
    maximum: float


PHYSICAL_RANGES: Dict[str, RangeRule] = {
    "met": RangeRule(0.5, 4.0),
    "clo": RangeRule(0.0, 3.0),
    "rh": RangeRule(0.0, 100.0),
    "vel": RangeRule(0.0, 3.0),
    "ta": RangeRule(0.0, 50.0),
    "tr": RangeRule(0.0, 50.0),
}


CLASS_LABELS: Tuple[str, str, str] = ("cold", "neutral", "hot")
