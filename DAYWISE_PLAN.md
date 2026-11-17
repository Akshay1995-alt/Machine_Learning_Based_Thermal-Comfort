# ASHRAE Thermal Comfort ML — 15-Day Learning & Execution Plan

This is a beginner-friendly, day-by-day roadmap for building your first end-to-end Machine Learning project on the **ASHRAE Thermal Comfort Database II**, designed around your CFD and thermodynamics background.

---

## Phase 1: Setup & Data Engineering (Days 1–4)

### Day 1: Environment Setup & Project Orientation
- **Notebook:** `notebooks/day_01_environment_setup.ipynb`
- **ML Concept:** Reproducible environments, virtual environments, and Jupyter kernels.
- **CFD Analogy:** Setting up your solver environment and case directory structure before meshing or running simulations.
- **Tasks:**
  1. Activate `.venv` and select it as the Jupyter kernel in VS Code.
  2. Place the ASHRAE Thermal Comfort Database II CSV in `data/raw/`.
  3. Verify imports (`pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`) and preview the first 5 rows.
- **Deliverable:** Working notebook connected to `.venv` that loads the raw dataset.

### Day 2: Problem Formulation & Target Engineering
- **Notebook:** `notebooks/day_02_problem_formulation.ipynb`
- **ML Concept:** Supervised learning, features (`X`) vs. target (`y`), classification vs. regression, and target discretization.
- **CFD Analogy:** Defining boundary conditions/inputs (`ta`, `rh`, `vel`, `tr`, `clo`, `met`) vs. the output field (`tsv` / comfort class) instead of solving Fanger's heat-balance equation analytically.
- **Tasks:**
  1. Inspect dataset shape, column names, data types, and summary statistics.
  2. Isolate the 6 core features (`ta`, `rh`, `vel`, `tr`, `clo`, `met`), target (`tsv`), and benchmark (`pmv`).
  3. Map `tsv` (-3 to +3) into 3 classes: `cold` ($\le -1$), `neutral` ($-0.5$ to $0.5$), `hot` ($\ge 1$), and check how many transition-band rows are dropped.
- **Deliverable:** Summary of inputs, class counts, and transition-band counts.

### Day 3: Data Cleaning & Physics-Informed Filtering
- **Notebook:** `notebooks/day_03_data_cleaning.ipynb`
- **ML Concept:** Data quality, missingness audit, outlier filtering, and domain-based sanity rules.
- **CFD Analogy:** Checking mesh quality and physical boundary validity (e.g., rejecting negative velocities or unphysical temperatures).
- **Tasks:**
  1. Count missing values per column.
  2. Apply physical bounds (`met` in [0.5, 4.0], `clo` in [0.0, 3.0], `rh` in [0, 100], `vel` in [0.0, 3.0], `ta` & `tr` in [0, 50]).
  3. Log row counts before and after each filter step.
- **Deliverable:** Clean data-quality audit table with zero physically impossible values.

### Day 4: Imputation & Feature Engineering
- **Notebook:** `notebooks/day_04_imputation_and_features.ipynb`
- **ML Concept:** Handling missing values, domain fallback vs. statistical imputation (`KNNImputer`, `SimpleImputer`), and avoiding data leakage.
- **CFD Analogy:** Approximating mean radiant temperature $T_r \approx T_a$ in well-mixed indoor spaces when radiation data is missing.
- **Tasks:**
  1. Fill missing `tr` with `ta` where available.
  2. Test `KNNImputer` vs. median imputation on remaining missing values.
  3. Optionally engineer simple physics features like operative temperature `t_op = 0.5 * (ta + tr)`.
  4. Save the cleaned dataset to `data/processed/`.
- **Deliverable:** Clean, complete feature table saved to `data/processed/`.

---

## Phase 2: Exploratory Data Analysis (Days 5–6)

### Day 5: Psychrometric EDA & Thermodynamic Boundaries
- **Notebook:** `notebooks/day_05_psychrometric_eda.ipynb`
- **ML Concept:** Visualizing feature-target relationships, class overlap, and feature correlation.
- **CFD Analogy:** Reading a psychrometric chart or contour plot to see where comfort zones lie in $(T_a, RH)$ space.
- **Tasks:**
  1. Plot `ta` vs. `rh` colored by comfort class (`cold`, `neutral`, `hot`).
  2. Plot `ta` vs. `vel` colored by comfort class to observe convective cooling effects.
  3. Plot a correlation heatmap of the 6 core features.
- **Deliverable:** Saved EDA plots in `reports/figures/`.

### Day 6: Categorical & Behavioral EDA (NV vs. AC)
- **Notebook:** `notebooks/day_06_categorical_eda.ipynb`
- **ML Concept:** Class imbalance, subgroup/domain shift, and stratified train/test splitting.
- **CFD Analogy:** Comparing HVAC-controlled rooms (AC) against Naturally Ventilated (NV) buildings where occupant adaptation shifts comfort ranges.
- **Tasks:**
  1. Plot class balance (`cold`, `neutral`, `hot`) and observe `neutral` dominance.
  2. Compare `ta`, `vel`, and `tsv` across AC vs. NV buildings using boxplots/violin plots (if building mode columns exist).
  3. Create a stratified 80/20 train/test split (`stratify=y`).
- **Deliverable:** Clear understanding of class imbalance and NV vs. AC shifts.

---

## Phase 3: Progressive Model Building (Days 7–10)

### Day 7: Baseline Modeling (Logistic Regression)
- **Notebook:** `notebooks/day_07_baseline_logreg.ipynb`
- **ML Concept:** Linear decision boundaries, feature scaling (`StandardScaler`), `Pipeline`, and baseline benchmarking.
- **CFD Analogy:** Running a fast, coarse-mesh linear baseline before higher-fidelity models.
- **Tasks:**
  1. Build a `Pipeline` with `StandardScaler` + `LogisticRegression`.
  2. Train on the training split and evaluate on the test split.
  3. Inspect accuracy vs. per-class recall.
- **Deliverable:** Baseline metrics recorded.

### Day 8: Handling Class Imbalance
- **Notebook:** `notebooks/day_08_handling_imbalance.ipynb`
- **ML Concept:** Cost-sensitive learning (`class_weight='balanced'`), precision-recall trade-offs, and sample weights.
- **CFD Analogy:** Weighting critical boundary-layer regions so minority phenomena aren't washed out by bulk flow.
- **Tasks:**
  1. Compare Logistic Regression without vs. with `class_weight='balanced'`.
  2. Observe changes in Macro F1 and `cold`/`hot` recall.
- **Deliverable:** Before/after comparison table showing why class balancing helps.

### Day 9: Tree-Based Models (Random Forest & XGBoost)
- **Notebook:** `notebooks/day_09_tree_models.ipynb`
- **ML Concept:** Decision trees, bagging (Random Forest), gradient boosting (XGBoost), and non-linear threshold learning.
- **CFD Analogy:** Capturing sharp regime transitions (e.g., convective cooling above $v > 0.5\text{ m/s}$).
- **Tasks:**
  1. Train `RandomForestClassifier` and `XGBClassifier`.
  2. Compare train vs. test Macro F1 to check overfitting.
  3. Compare against Logistic Regression.
- **Deliverable:** Trained tree models and Macro F1 comparison.

### Day 10: Deep Learning Intro (MLP)
- **Notebook:** `notebooks/day_10_deep_learning_mlp.ipynb`
- **ML Concept:** Multi-Layer Perceptron (neural network), hidden layers, activation functions, loss curves, and early stopping.
- **CFD Analogy:** Universal function approximators learning coupled multi-physics mappings from data.
- **Tasks:**
  1. Build a `Pipeline` with `StandardScaler` + `MLPClassifier`.
  2. Plot the training loss curve.
  3. Compare MLP performance against Random Forest and XGBoost.
- **Deliverable:** Converged MLP baseline and tabular ML vs. DL comparison.

---

## Phase 4: Evaluation, Tuning & Interpretability (Days 11–13)

### Day 11: Evaluation Metrics & Fanger PMV Benchmark
- **Notebook:** `notebooks/day_11_evaluation_and_pmv.ipynb`
- **ML Concept:** Confusion matrices, Macro F1 vs. Accuracy, and domain baseline benchmarking.
- **CFD Analogy:** Validating a new model against classical analytical/empirical correlations (Fanger's PMV).
- **Tasks:**
  1. Generate confusion matrices and `classification_report` for all models.
  2. Map `pmv` to the same 3 classes (`cold`, `neutral`, `hot`).
  3. Compare ML Macro F1 directly against Fanger PMV Macro F1.
- **Deliverable:** Full benchmark table and confusion matrix heatmaps.

### Day 12: Hyperparameter Tuning
- **Notebook:** `notebooks/day_12_hyperparameter_tuning.ipynb`
- **ML Concept:** Hyperparameters, k-fold cross-validation (`StratifiedKFold`), `GridSearchCV`, and `Optuna`.
- **CFD Analogy:** Running a systematic parametric study / DoE on solver settings using validation metrics.
- **Tasks:**
  1. Run `GridSearchCV` on Random Forest (and optionally `Optuna` on XGBoost).
  2. Inspect best parameters and cross-validation Macro F1.
  3. Evaluate the final tuned model once on the held-out test set.
- **Deliverable:** Tuned model and CV vs. test comparison.

### Day 13: Model Interpretability (Feature Importance & SHAP)
- **Notebook:** `notebooks/day_13_interpretability_shap.ipynb`
- **ML Concept:** Global feature importance, Shapley values (SHAP), and physics sanity-checking.
- **CFD Analogy:** Post-processing sensitivity derivatives to confirm the model obeys physical intuition.
- **Tasks:**
  1. Plot tree feature importances (`ta`, `tr`, `vel`, `rh`, `clo`, `met`).
  2. Generate SHAP summary plot and individual waterfall explanations.
  3. Write a physics-based interpretation of the top drivers.
- **Deliverable:** Feature importance + SHAP visuals with engineering takeaways.

---

## Phase 5: CFD Integration & Spatial Mapping (Days 14–15)

### Day 14: CFD Integration Prep
- **Notebook:** `notebooks/day_14_cfd_integration_prep.ipynb`
- **ML Concept:** Batch inference, schema/unit alignment, and array formatting.
- **CFD Analogy:** Exporting cell-center or nodal field data `(x, y, z, ta, vel, ...)` from Fluent / OpenFOAM / ParaView.
- **Tasks:**
  1. Load a CFD field CSV export.
  2. Check units (e.g., Kelvin to °C) and fill assumed boundary parameters (`rh`, `clo`, `met`, `tr`).
  3. Format `X_cfd` with exact columns `["ta", "rh", "vel", "tr", "clo", "met"]`.
- **Deliverable:** Validated `X_cfd` feature matrix ready for prediction.

### Day 15: Spatial Comfort Mapping & Portfolio Wrap-Up
- **Notebook:** `notebooks/day_15_spatial_comfort_mapping.ipynb`
- **ML Concept:** Batch prediction (`predict`, `predict_proba`) and end-to-end deployment.
- **CFD Analogy:** Computing a custom scalar field across the 3D domain and rendering comfort contours in ParaView.
- **Tasks:**
  1. Run your best trained model on `X_cfd` to get predicted classes and `P(neutral)`.
  2. Export `(x, y, z, predicted_class, comfort_prob)` to CSV for ParaView visualization.
  3. Summarize final results in `README.md`.
- **Deliverable:** 3D spatial comfort contour CSV/visual + portfolio-ready project.
