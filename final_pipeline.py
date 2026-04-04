# ===== IMPORTS =====
from stage_a_baseline import preprocess_data, train_model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ===== STEP 1: LOAD DATA =====
df = pd.read_csv("KDDTrain_with_headers.csv")


# ===== STEP 2: RAW DATA (FOR STAGE C) =====
X_raw = df.drop(columns=["label"])
y_raw = df["label"]

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)


# ===== STEP 3: PROCESSED DATA (FOR STAGE A) =====
X_train, X_test, y_train, y_test = preprocess_data(df)


# ===== STEP 4: TRAIN STAGE A MODEL =====
clf_model = train_model(X_train, y_train, "rf")

# Stage A predictions
clf_pred = clf_model.predict(X_test)


# ===== STEP 5: LOAD STAGE C MODEL =====
iso_model = joblib.load("stage_c_isolation_forest.pkl")

# Prepare numeric-only data for Stage C
X_test_numeric = X_test_raw.select_dtypes(include=[np.number])

# Match feature names exactly
if hasattr(iso_model, "feature_names_in_"):
    X_test_numeric = X_test_numeric[iso_model.feature_names_in_]

# Stage C predictions
anomaly_pred = iso_model.predict(X_test_numeric)


# ===== STEP 6: FINAL DECISION (YOUR ROLE 🔥) =====
def final_decision(clf_pred, anomaly_pred):
    final = []
    for c, a in zip(clf_pred, anomaly_pred):
        if c == 1 or a == -1:
            final.append(1)  # attack
        else:
            final.append(0)  # normal
    return np.array(final)


final_pred = final_decision(clf_pred, anomaly_pred)


# ===== STEP 7: CONVERT TRUE LABELS TO BINARY =====
y_test_binary = (y_test_raw.astype(str).str.lower() != "normal").astype(int)


# ===== STEP 8: PRINT RESULTS =====
print("\nFinal Predictions (first 20):")
print(final_pred[:20])


# ===== STEP 9: METRICS =====
print("\nClassification Report:")
print(classification_report(y_test_binary, final_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_binary, final_pred))


# ===== STEP 10: COMPARE STAGE A vs FINAL =====
print("\nStage A vs Final Comparison:")
print("Stage A Accuracy:", accuracy_score(y_test_binary, clf_pred))
print("Final Accuracy:", accuracy_score(y_test_binary, final_pred))


# ===== STEP 11: VISUALIZATION =====
ConfusionMatrixDisplay.from_predictions(y_test_binary, final_pred)
plt.title("Final IDS Confusion Matrix")
plt.show()


# ===== STEP 12: SAVE OUTPUT =====
df_out = pd.DataFrame({
    "Actual": y_test_binary,
    "Predicted": final_pred
})

df_out.to_csv("final_output_with_labels.csv", index=False)