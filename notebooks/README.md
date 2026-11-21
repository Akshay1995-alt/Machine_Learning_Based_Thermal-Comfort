# Project 1 Notebooks

Complete these notebooks in order for the global thermal-sensation
regression project. They are written for a first machine-learning project,
with explanatory Markdown and small, readable code steps.

1. `01_load_and_clean.ipynb` — Load the raw measurements and metadata, inspect
   the data, validate the building join, and prepare the target-specific data.
2. `02_eda_feature_engineering.ipynb` — Explore the training data, select
   candidate features, and create a building-based train/test split.
3. `03_baseline_model.ipynb` — Establish simple regression baselines and use
   a preprocessing pipeline with grouped cross-validation.
4. `04_model_evaluation.ipynb` — Compare models, evaluate on held-out
   buildings, and inspect prediction errors.
5. `05_feature_importance_and_shap.ipynb` — Interpret the selected model with
   permutation feature importance and SHAP.

The original CSV files belong in `data/raw/`. Do not commit the data files.
See the repository-root `DAYWISE_PLAN.md` for the overall sequence and
Project 2 plan.
