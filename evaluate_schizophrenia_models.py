"""
Model Evaluation Pipeline for Schizophrenia Synthetic Dataset

Compares classical statistical models vs ML models:
  1. Logistic Regression (top-ranked features)
  2. Logistic Regression (all features)
  3. Random Forest
  4. Support Vector Machine (RBF kernel)
  5. XGBoost (gradient boosting)
  6. Neural Network (MLP)

Reports AUC, accuracy, sensitivity, specificity, F1 for each model
using 5-fold stratified cross-validation with proper imputation/scaling.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, confusion_matrix, make_scorer,
)
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv("schizophrenia_synthetic_dataset.csv")

# Drop identifier and string variant-ID columns
genes = ["C4A", "DRD2", "GRM3", "GRIN2A", "SLC39A8", "DISC1", "COMT", "BDNF"]
drop_cols = ["subject_id"] + [f"{g}_variant_id" for g in genes]
df = df.drop(columns=drop_cols)

y = df["diagnosis"].values
X = df.drop(columns=["diagnosis"])
feature_names = X.columns.tolist()

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class balance: {(y==1).sum()} cases / {(y==0).sum()} controls\n")

# ============================================================================
# Feature ranking (mutual information + logistic regression coefficients)
# ============================================================================

print("=" * 70)
print("FEATURE RANKING")
print("=" * 70)

X_imp = SimpleImputer(strategy="median").fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imp)

# Mutual information
mi_scores = mutual_info_classif(X_imp, y, random_state=SEED)
mi_df = pd.DataFrame({"feature": feature_names, "MI": mi_scores})

# Logistic regression absolute coefficients
lr_full = LogisticRegression(max_iter=2000, random_state=SEED, penalty="l2", C=1.0)
lr_full.fit(X_scaled, y)
coef_df = pd.DataFrame({
    "feature": feature_names,
    "LR_coef": lr_full.coef_[0],
    "|LR_coef|": np.abs(lr_full.coef_[0]),
})

# Merge and rank
rank_df = mi_df.merge(coef_df[["feature", "LR_coef", "|LR_coef|"]], on="feature")
rank_df["MI_rank"] = rank_df["MI"].rank(ascending=False).astype(int)
rank_df["LR_rank"] = rank_df["|LR_coef|"].rank(ascending=False).astype(int)
rank_df["avg_rank"] = (rank_df["MI_rank"] + rank_df["LR_rank"]) / 2
rank_df = rank_df.sort_values("avg_rank").reset_index(drop=True)

print(f"\n{'Rank':>4s}  {'Feature':>24s}  {'MI':>6s}  {'MI_rk':>5s}  "
      f"{'LR_coef':>8s}  {'LR_rk':>5s}  {'Avg_rk':>6s}")
print("-" * 70)
for i, row in rank_df.head(20).iterrows():
    print(f"{i+1:4d}  {row['feature']:>24s}  {row['MI']:6.4f}  "
          f"{int(row['MI_rank']):5d}  {row['LR_coef']:8.4f}  "
          f"{int(row['LR_rank']):5d}  {row['avg_rank']:6.1f}")

top_features = rank_df.head(10)["feature"].tolist()
print(f"\nTop 10 features selected for reduced logistic regression:")
for i, f in enumerate(top_features, 1):
    print(f"  {i:2d}. {f}")

# ============================================================================
# Custom scorers
# ============================================================================

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scoring = {
    "AUC": "roc_auc",
    "accuracy": "accuracy",
    "sensitivity": make_scorer(recall_score),
    "specificity": make_scorer(specificity_score),
    "precision": "precision",
    "F1": "f1",
}

# ============================================================================
# Define models
# ============================================================================

models = {
    "Logistic Reg. (top 10)": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED,
                                       penalty="l2", C=1.0)),
        ]),
        "features": top_features,
    },
    "Logistic Reg. (all)": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED,
                                       penalty="l2", C=1.0)),
        ]),
        "features": feature_names,
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=3,
                max_features="sqrt",
                random_state=SEED, n_jobs=-1)),
        ]),
        "features": feature_names,
    },
    "SVM (RBF)": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                        probability=True, random_state=SEED)),
        ]),
        "features": feature_names,
    },
    "XGBoost": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                min_child_weight=3, reg_alpha=0.1,
                random_state=SEED, eval_metric="logloss",
                verbosity=0, n_jobs=-1)),
        ]),
        "features": feature_names,
    },
    "Neural Network (MLP)": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                solver="adam", max_iter=800, random_state=SEED,
                early_stopping=True, validation_fraction=0.15,
                learning_rate="adaptive", alpha=0.001)),
        ]),
        "features": feature_names,
    },
}

# ============================================================================
# 5-fold stratified cross-validation
# ============================================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON — 5-Fold Stratified Cross-Validation")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = {}

for name, spec in models.items():
    feats = spec["features"]
    X_sub = X[feats]
    pipe = spec["pipeline"]

    cv_results = cross_validate(
        pipe, X_sub, y, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1,
    )

    res = {}
    for metric_key in scoring:
        vals = cv_results[f"test_{metric_key}"]
        res[metric_key] = (vals.mean(), vals.std())
    results[name] = res

# Print results table
print(f"\n{'Model':>25s} | {'AUC':>12s} | {'Accuracy':>12s} | "
      f"{'Sensitivity':>12s} | {'Specificity':>12s} | {'F1':>12s}")
print("-" * 100)

for name, res in results.items():
    auc_m, auc_s = res["AUC"]
    acc_m, acc_s = res["accuracy"]
    sen_m, sen_s = res["sensitivity"]
    spe_m, spe_s = res["specificity"]
    f1_m, f1_s = res["F1"]
    print(f"{name:>25s} | {auc_m:.3f}±{auc_s:.3f} | {acc_m:.3f}±{acc_s:.3f} | "
          f"{sen_m:.3f}±{sen_s:.3f} | {spe_m:.3f}±{spe_s:.3f} | {f1_m:.3f}±{f1_s:.3f}")

# ============================================================================
# Best model identification
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Sort all models by AUC
sorted_models = sorted(results.items(), key=lambda kv: kv[1]["AUC"][0], reverse=True)

print("\nModels ranked by AUC:\n")
for rank, (name, res) in enumerate(sorted_models, 1):
    auc_m, auc_s = res["AUC"]
    sen_m, _ = res["sensitivity"]
    spe_m, _ = res["specificity"]
    print(f"  {rank}. {name:<28s}  AUC={auc_m:.4f} ± {auc_s:.4f}  "
          f"Sens={sen_m:.3f}  Spec={spe_m:.3f}")

best_name = sorted_models[0][0]
best_auc = sorted_models[0][1]["AUC"][0]

lr_top10_auc = results["Logistic Reg. (top 10)"]["AUC"][0]
lr_all_auc = results["Logistic Reg. (all)"]["AUC"][0]
best_ml_name = next(n for n, _ in sorted_models
                    if "Logistic" not in n)
best_ml_auc = results[best_ml_name]["AUC"][0]

print(f"\n{'Logistic Regression (top 10) AUC:':>40s} {lr_top10_auc:.4f}")
print(f"{'Logistic Regression (all feats) AUC:':>40s} {lr_all_auc:.4f}")
print(f"{'Best ML model AUC:':>40s} {best_ml_auc:.4f}  ({best_ml_name})")

delta = best_ml_auc - lr_all_auc
if delta > 0.005:
    print(f"\n→ ML advantage over logistic regression: +{delta:.4f} AUC")
    print(f"  {best_ml_name} outperforms logistic regression, suggesting")
    print(f"  nonlinear feature interactions contribute to prediction.")
elif delta > -0.005:
    print(f"\n→ ML and logistic regression perform comparably (Δ AUC = {delta:+.4f})")
    print(f"  The decision boundary is largely linear; ML models capture")
    print(f"  only marginal additional signal.")
else:
    print(f"\n→ Logistic regression outperforms the best ML model by {-delta:.4f} AUC")
    print(f"  The problem is well-suited to linear modeling.")

# ============================================================================
# Random Forest feature importances
# ============================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST — Feature Importances (Gini)")
print("=" * 70)

rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1)),
])
rf_pipe.fit(X, y)
rf_clf = rf_pipe.named_steps["clf"]
imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_clf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n{'Rank':>4s}  {'Feature':>24s}  {'Importance':>10s}")
print("-" * 50)
for i, row in imp_df.head(20).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"{i+1:4d}  {row['feature']:>24s}  {row['importance']:10.4f}  {bar}")

# ============================================================================
# XGBoost feature importances (gain)
# ============================================================================

print("\n" + "=" * 70)
print("XGBOOST — Feature Importances (Gain)")
print("=" * 70)

xgb_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric="logloss",
        verbosity=0, n_jobs=-1)),
])
xgb_pipe.fit(X, y)
xgb_clf = xgb_pipe.named_steps["clf"]
xgb_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb_clf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n{'Rank':>4s}  {'Feature':>24s}  {'Importance':>10s}")
print("-" * 50)
for i, row in xgb_imp.head(20).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"{i+1:4d}  {row['feature']:>24s}  {row['importance']:10.4f}  {bar}")

print("\nDone.")
