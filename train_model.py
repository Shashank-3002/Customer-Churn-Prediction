import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Loading Dataset...")

df = pd.read_csv("Churn_Modelling.csv")

print(df.head())

df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

df = pd.get_dummies(
    df,
    columns=["Geography"],
    drop_first=True
)

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeature Columns:")
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========\n")

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop Feature Importance:")
print(feature_importance)

# Optional Visualization
plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind="barh")
plt.title("Top Features Affecting Churn")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

print("\nSaving Model Files...")

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel + Scaler Saved Successfully!")
print("Ready for Streamlit Deployment 🚀")