# ============================================================
#  TRAIN MODEL  —  run this once before launching the app
#  pip install xgboost shap imbalanced-learn scikit-learn pandas joblib matplotlib
#  python train_model.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve,
)

print("\n" + "="*55)
print("  CHURN MODEL TRAINING")
print("="*55)

# ── 1. LOAD ──────────────────────────────────────────────────
df = pd.read_csv("Churn_Modelling.csv")
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]
FEATURES = list(X.columns)
print(f"\n  Dataset : {df.shape[0]:,} rows | Churn rate: {y.mean():.1%}")

# ── 2. SPLIT ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. SCALE ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 4. SMOTE (fix class imbalance) ───────────────────────────
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_sc, y_train)
print(f"  After SMOTE : {len(X_res):,} training samples (balanced)")

# ── 5. TRAIN XGBoost ─────────────────────────────────────────
print("\n  Training XGBoost ...")
model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="auc",
    random_state=42, verbosity=0,
)
model.fit(X_res, y_res, eval_set=[(X_test_sc, y_test)], verbose=False)

# ── 6. TUNE THRESHOLD (maximise F1) ──────────────────────────
y_prob = model.predict_proba(X_test_sc)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, y_prob)
f1s  = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f1s[:-1])])
y_pred   = (y_prob >= best_thr).astype(int)

# ── 7. RESULTS ───────────────────────────────────────────────
print("\n" + "="*55)
print("  EVALUATION")
print("="*55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  Threshold : {best_thr:.3f}  (F1-optimal)")
print(f"\n{classification_report(y_test, y_pred)}")

# ── 8. SHAP ──────────────────────────────────────────────────
print("  Computing SHAP values ...")

# XGBoost 2.x / SHAP compatibility fix
model.save_model("_tmp_model.json")
import xgboost as xgb
_m = xgb.XGBClassifier(); _m.load_model("_tmp_model.json")
explainer = shap.TreeExplainer(_m)

shap_vals = explainer.shap_values(X_test_sc)

plt.figure(figsize=(9, 5))
shap.summary_plot(shap_vals, X_test_sc, feature_names=FEATURES,
                  plot_type="bar", show=False)
plt.title("Feature Importance (SHAP)", fontsize=12)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()
print("  SHAP plot saved → shap_importance.png")

# ── 9. SAVE ──────────────────────────────────────────────────
joblib.dump(model,     "churn_model.pkl")
joblib.dump(scaler,    "scaler.pkl")
joblib.dump(explainer, "shap_explainer.pkl")
joblib.dump(FEATURES,  "feature_names.pkl")
joblib.dump(best_thr,  "threshold.pkl")

import os; os.remove("_tmp_model.json")

print("\n" + "="*55)
print("  FILES SAVED")
print("="*55)
print("  churn_model.pkl")
print("  scaler.pkl")
print("  shap_explainer.pkl")
print("  feature_names.pkl")
print("  threshold.pkl")
print("\n  ✅  Run:  streamlit run app.py")
print("="*55 + "\n")