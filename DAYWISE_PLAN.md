# Comfort Cooling Project Plan

## Project overview

This project contains two complementary machine-learning studies using the
ASHRAE Global Thermal Comfort Database II:

1. **Thermal Sensation:** predict how an occupant feels, from cold to hot.
2. **Thermal Preference:** predict whether an occupant prefers cooler
   conditions, no change, or warmer conditions.

The projects will be completed **sequentially**, not in parallel. We are
currently working only on Project 1, planned for 8 study days. After Project 1
is complete, Project 2 will take 5 study days and its data preparation will
start again from the original CSV files. Project 2 will have its own cleaned
data, checks, and analysis; it will not use Project 1's cleaned dataset.

**Geographic scope:** Both projects use the global dataset; do not filter to the
UK or another single country for the primary analysis. Preserve country and
climate information for context and later comparison of model performance
across locations.

## Data location

The raw dataset files are stored in `data/raw/`:

- `Ashrae_Comfort_Cooling_Database_v2.1.0.csv` — measurements
- `Ashrae_Comfort_Cooling_Database_Metadata.csv` — building and study metadata

Keep the raw files unchanged. Store any generated, cleaned data in
`data/processed/`. Record filters and transformations so the analysis can be
reproduced.

## Project principles

- Keep the code beginner-friendly: use clear variable names, small cells, and
  Markdown explanations before each major step. Add concise comments to explain
  non-obvious choices and checks, rather than commenting every line.
- Prefer straightforward Python and pandas patterns. Introduce abstractions or
  advanced techniques only when they help the project, and explain them when
  they are introduced.
- Inspect the specific database version and codebook before relying on
  column names, units, or vote coding.
- Check target coverage and class balance before finalizing either project.
- Join metadata to measurements using `building_id` only when needed, and
  verify the join does not duplicate measurement records.
- Prefer a grouped test split that holds out entire buildings or publications.
  A random row split alone may put related observations in both training and
  test sets.
- Fit imputation, encoding, scaling, feature selection, and other learned
  preprocessing on training data only, using a scikit-learn pipeline.
- Do not use other occupant-reported votes or calculated comfort indices
  (such as PMV, PPD, or SET) as predictors of the occupant-reported target.
- SHAP and feature-importance results describe model associations; they do not
  establish causality.

## Project 1: Thermal Sensation — 8 days

**Question:** Can indoor conditions and building context predict an
occupant's thermal sensation?

**Target:** `thermal_sensation`, described in the database documentation as a
vote from -3 (cold) through 0 (neutral) to +3 (hot). The inspected file also
contains fractional values, so use numeric regression as the primary
formulation and preserve the original values. A classification analysis can
be added only with a documented, justified mapping from numeric votes to
categories.

| Day | Category | Activities and deliverable |
|---|---|---|
| 1 | Data loading and cleaning | Load both CSVs, inspect their structure and target, standardize text formatting, check missing values and repeated rows, validate the `building_id` join, and create a target-specific cleaned table without modifying raw files. Remove rows only when the target is missing; do not impute feature values or delete repeated measurements just because values match. **Deliverable:** reproducible cleaned thermal-sensation dataset and data-quality notes. |
| 2 | EDA and feature engineering | Make a grouped train/test split by building before making modeling choices. Explore target and candidate-variable distributions in the training data, inspect missingness, choose initial predictors, and explain excluded columns. **Deliverable:** EDA notebook, feature list, and group split plan. |
| 3 | Baseline model | Build a straightforward baseline and a simple regression model using a scikit-learn preprocessing pipeline. **Deliverable:** first model and baseline metrics. |
| 4 | Additional model creation | Train a tree-based regression model such as Random Forest or gradient boosting and compare it with the baseline. **Deliverable:** comparable candidate models. |
| 5 | Evaluation and error analysis | Compare models on the same held-out groups using MAE, RMSE, and R²; inspect residuals and errors across the sensation range. **Deliverable:** model comparison and error review. |
| 6 | Improve the model carefully | Test a small number of justified changes to features or model settings, using training data or cross-validation for decisions. Keep the final test set untouched. **Deliverable:** selected model with a fair evaluation. |
| 7 | Feature importance and SHAP | Calculate permutation feature importance on held-out data. Apply SHAP to the selected model to examine overall patterns and selected individual predictions. **Deliverable:** interpretation plots and caveats. |
| 8 | Communication and reproducibility | Organize notebooks, move stable reusable code into `src/comfort_cooling/`, and write a short report describing the question, data, method, results, limitations, and next steps. **Deliverable:** reproducible project and report. |

### Project 1 evaluation

For the numeric target, report MAE, RMSE, R², and residual/error summaries
across the vote range. Do not round fractional votes silently. If you also
create a classification version, explain its mapping and report macro F1,
balanced accuracy, and a confusion matrix; do not merge categories simply to
improve metrics.

## Project 2: Thermal Preference — 5 days

**Status:** Start this project only after Project 1 is complete. Begin its data
preparation from both original CSV files in `data/raw/`. Repeat loading,
inspection, cleaning, and join validation for the new target. Do not use
Project 1's cleaned CSV as the input to Project 2. This deliberate restart lets
you practice the data-preparation workflow a second time and keeps each
target's cleaning decisions independent.

**Question:** Can the same types of inputs predict whether an occupant prefers
cooler conditions, no change, or warmer conditions?

**Target:** `thermal_preference`. Audit its own missingness, label values, and
class balance directly from the raw measurements. Follow the same careful
principles as Project 1, but make and document target-specific cleaning
decisions independently.

| Day | Category | Activities and deliverable |
|---|---|---|
| 1 | Data loading and cleaning | Start again from the two raw CSVs. Load and inspect them, clean text formatting, check missing values and repeated rows, validate the `building_id` join, and create a preference-specific table by excluding only rows without this target. Do not use Project 1's processed data. **Deliverable:** independently prepared preference dataset and target audit. |
| 2 | EDA and feature engineering | Make a grouped split by building, explore preference in training data, and define a comparable feature set. **Deliverable:** preference EDA, feature list, and split plan. |
| 3 | ML model creation | Train a majority-class baseline, Logistic Regression, and the strongest model from Project 1 in an appropriate classification pipeline. **Deliverable:** working preference models. |
| 4 | Evaluation, feature importance, and SHAP | Evaluate using macro F1, balanced accuracy, a confusion matrix, and per-class performance. Apply permutation importance, then SHAP to the best model. **Deliverable:** performance and interpretation results. |
| 5 | Combined interpretation and communication | Compare the two projects' predictions on the same held-out observations. Investigate cases where sensation and preference differ. Create a simple combined result view and document limitations. **Deliverable:** paired-project summary. |

### Project 2 evaluation

Report macro F1, balanced accuracy, a confusion matrix, and per-class
performance. Keep these results separate from the Thermal Sensation scores:
the targets have different meanings, so their metrics are not directly
comparable.

## Combined project interpretation

The two outputs describe different aspects of occupant response:

- `thermal_sensation`: how the occupant feels now.
- `thermal_preference`: what change the occupant would like.

Showing both can provide a richer comfort-response summary. For example, an
occupant might feel slightly warm yet prefer no change, or feel neutral while
preferring cooler conditions. Analyze such cases rather than assuming one vote
determines the other.

## Limitations

The database contains observational field measurements and occupant votes.
Predictions and what-if plots are not proof that changing a setpoint or fan
speed causes a particular comfort outcome. The described database does not
provide direct HVAC energy-use measurements, so these projects should not claim
measured energy savings.
