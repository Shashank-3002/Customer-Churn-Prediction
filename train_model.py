# train_model.py (Deployment Safe Training File)

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("Churn_Modelling.csv")

print(df.head())

# -----------------------------------
# Data Preprocessing
# -----------------------------------

# Remove unnecessary columns
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

# -----------------------------------
# Train Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# Feature Scaling
# -----------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Model Training
# -----------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# Evaluation
# -----------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------
# Save Files
# -----------------------------------
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel + Scaler Saved Successfully!")