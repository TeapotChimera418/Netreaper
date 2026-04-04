import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

# Load Data
df = pd.read_csv(r"F:\NetReaper\KDDTrain_with_headers.csv")

if "difficulty" in df.columns:
    df = df.drop(columns=["difficulty"])


# Preprocessor function
def preprocess_data(df):
    data = df.copy()

    # Binary label
    y = (data["label"].astype(str).str.lower() != "normal").astype(int)
    X = data.drop(columns=["label"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test


# Random forest
def train_model(X_train, y_train, model_type):
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


# Start of the pipeline
X_train, X_test, y_train, y_test = preprocess_data(df)

# Stage A
clf_model = train_model(X_train, y_train, "rf")
clf_pred = clf_model.predict(X_test)

# Stage B - Attack
X_adv = X_test + np.random.normal(0.7, 1.3, X_test.shape)
clf_pred_adv = clf_model.predict(X_adv)

# Stage C - Anomaly
iso_model = joblib.load(r"F:\NetReaper\stage_c_isolation_forest.pkl")

X_test_for_iso = pd.DataFrame(X_test)

if hasattr(iso_model, "feature_names_in_"):
    missing_cols = set(iso_model.feature_names_in_) - set(X_test_for_iso.columns)
    for col in missing_cols:
        X_test_for_iso[col] = 0
    X_test_for_iso = X_test_for_iso[iso_model.feature_names_in_]

anomaly_pred = iso_model.predict(X_test_for_iso)

#Final Logic
final_pred = clf_pred_adv.copy()
final_pred[anomaly_pred == -1] = 1

# Results
print("\n=== RESULTS ===")

print("\nStage A Accuracy:")
print(accuracy_score(y_test, clf_pred))

print("\nAfter Attack Accuracy:")
print(accuracy_score(y_test, clf_pred_adv))

print("\nFinal Accuracy:")
print(accuracy_score(y_test, final_pred))

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))

# Visulaization
ConfusionMatrixDisplay.from_predictions(y_test, final_pred)
plt.title("Final IDS Confusion Matrix")
plt.show()

# Save Output
df_out = pd.DataFrame({
    "Actual": y_test,
    "Predicted": final_pred
})

df_out.to_csv("final_output_with_labels.csv", index=False)
