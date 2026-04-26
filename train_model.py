import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

print("\n" + "=" * 55)
print("  CHURN MODEL TRAINING")
print("=" * 55)

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

# Encode Gender
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# One-Hot Encode Geography
df = pd.get_dummies(
    df,
    columns=["Geography"],
    drop_first=True
)

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

FEATURES = list(X.columns)

print(f"\nDataset Shape: {df.shape}")
print(f"Churn Rate: {y.mean():.2%}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Class Imbalance using SMOTE
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

print(f"Balanced Training Samples: {len(X_resampled)}")

# Train XGBoost Model
print("\nTraining XGBoost Model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    verbosity=0
)

model.fit(
    X_resampled,
    y_resampled
)

# Prediction Probabilities
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Threshold Optimization using F1 Score
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

f1_scores = 2 * precision * recall / (
    precision + recall + 1e-9
)

best_threshold = float(
    thresholds[np.argmax(f1_scores[:-1])]
)

y_pred = (y_prob >= best_threshold).astype(int)

# Evaluation
print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"Accuracy Score : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score  : {roc_auc_score(y_test, y_prob):.4f}")
print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")
print(f"Best Threshold : {best_threshold:.3f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Static Feature Importance Plot
feature_importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind="barh")
plt.title("Top Features Affecting Customer Churn")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nFeature importance plot saved → feature_importance.png")

# Save Files for Streamlit Deployment
print("\nSaving Files...")

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_names.pkl")
joblib.dump(best_threshold, "threshold.pkl")

print("\n" + "=" * 55)
print("FILES SAVED SUCCESSFULLY")
print("=" * 55)
print("churn_model.pkl")
print("scaler.pkl")
print("feature_names.pkl")
print("threshold.pkl")
print("feature_importance.png")
print("\n✅ Ready for Streamlit Deployment")
print("=" * 55)