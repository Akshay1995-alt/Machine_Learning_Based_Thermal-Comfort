# ASHRAE Thermal Comfort ML (Phase 1)

This project builds a 3-class thermal sensation classifier from ASHRAE Thermal Comfort Database II style data.

## Phase 1 scope
- Task: classification (`cold`, `neutral`, `hot`)
- Models: Logistic Regression, Random Forest, XGBoost (if available)
- Metrics: Macro F1, per-class precision/recall/F1, confusion matrices
- Benchmark: PMV-threshold baseline comparison (if PMV column exists)
- Day-by-day learning guide: [DAYWISE_PLAN.md](DAYWISE_PLAN.md)

## Folder layout
```text
ashrae-thermal-comfort-ml/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ models/
├─ notebooks/
├─ reports/
│  └─ figures/
├─ src/
│  └─ ashrae_ml/
└─ README.md
```

## Setup
From [Machine_Learning_Projects](C:/Akshay/Machine_Learning_Projects):

```powershell
.\.venv\Scripts\Activate.ps1
cd .\ashrae-thermal-comfort-ml
pip install -r requirements.txt
pip install -e .
```

## Expected dataset columns
Core features:
- `ta` (air temperature)
- `rh` (relative humidity)
- `vel` (air velocity)
- `tr` (mean radiant temperature)
- `clo` (clothing insulation)
- `met` (metabolic rate)

Target:
- `tsv` thermal sensation vote in `[-3, 3]`

Optional PMV benchmark column:
- `pmv` (default name, configurable)

## Class mapping used
- `cold`: TSV <= -1
- `neutral`: -0.5 <= TSV <= 0.5
- `hot`: TSV >= 1

Rows in transition bands `(-1, -0.5)` and `(0.5, 1)` are dropped from supervised training/evaluation.

## Data quality rules
- Keep only physically plausible ranges:
  - `met` in [0.5, 4.0]
  - `clo` in [0.0, 3.0]
  - `rh` in [0, 100]
  - `vel` in [0.0, 3.0]
  - `ta` in [0, 50]
  - `tr` in [0, 50]
- Missing `tr` handling:
  1) Domain-informed fallback: if `tr` missing and `ta` present, set `tr = ta`
  2) Median imputation for remaining missing values

## Run training
```powershell
python -m ashrae_ml.train --data .\data\raw\ashrae_db2.csv --target-col tsv --pmv-col pmv
```

Outputs:
- `models/*.pkl`: trained pipelines
- `reports/metrics_summary.csv`: model metrics
- `reports/confusion_matrix_*.csv`
- `reports/classification_report_*.csv`
- `reports/pmv_baseline_metrics.csv` (if PMV column available)

## Run starter EDA
```powershell
python -m ashrae_ml.eda --data .\data\raw\ashrae_db2.csv --target-col tsv
```

Outputs:
- `reports/figures/class_balance.png`
- `reports/figures/ta_rh_scatter_by_class.png`

## Interpretability
```powershell
python -m ashrae_ml.interpret --data .\data\raw\ashrae_db2.csv --model-path .\models\random_forest.pkl
```

Outputs:
- `reports/feature_importance.csv`
- `reports/shap_summary.csv` (for tree-compatible models when SHAP succeeds)

## Hyperparameter tuning (Random Forest)
```powershell
python -m ashrae_ml.tune --data .\data\raw\ashrae_db2.csv
```

Outputs:
- `reports/rf_gridsearch_results.csv`
- `models/random_forest_tuned.pkl`

