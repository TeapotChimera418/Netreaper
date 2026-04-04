import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from stage_a_baseline import preprocess_data, train_model

# 1. Loading the data
df = pd.read_csv(r"KDDTrain_with_headers.csv")

# 2. Processed Data (Unified Split)
# This ensures X_test and y_test are perfectly aligned (0=Normal, 1=Attack)
X_train, X_test, y_train, y_test = preprocess_data(df)

# 3. Running Stage A (Baseline Classifier)
print("--- Running Stage A (Baseline Classifier) ---")
clf_model = train_model(X_train, y_train, "rf")
clf_pred = clf_model.predict(X_test)

# 4. Running Stage C (Anomaly Detection)
print("--- Running Stage C (Anomaly Detection) ---")
iso_model = joblib.load(r"stage_c_isolation_forest.pkl")

# Prepare numeric columns from the processed X_test
X_test_numeric = X_test.select_dtypes(include=[np.number]).copy()

# CLEAN PREFIXES: Strip 'num__' so names match your 38-feature Stage C model
X_test_numeric.columns = [c.replace('num__', '').replace('cat__', '') for c in X_test_numeric.columns]

# ALIGN FEATURES: Ensure 'difficulty' is dropped and order is correct
if hasattr(iso_model, "feature_names_in_"):
    X_test_numeric = X_test_numeric[iso_model.feature_names_in_]

# Stage C predictions (-1 for Anomaly, 1 for Normal)
anomaly_pred = iso_model.predict(X_test_numeric)

# 5. Final Resilient Decision Logic
def final_decision(clf_preds, anomaly_preds):
    final = []
    for c, a in zip(clf_preds, anomaly_preds):
        if c == 1 or a == -1:
            final.append(1)  # Flag as Attack
        else:
            final.append(0)  # Flag as Normal
    return np.array(final)

final_pred = final_decision(clf_pred, anomaly_pred)

# 6. Evaluation & Metrics
print("\n" + "="*30)
print("FINAL PIPELINE RESULTS")
print("="*30)

print(f"Stage A Accuracy: {accuracy_score(y_test, clf_pred):.4f}")
print(f"Final Accuracy:   {accuracy_score(y_test, final_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, final_pred, target_names=["Normal", "Attack"]))

# 7. VISUALIZATION (Requirement: Picture Output)
# Generates the Confusion Matrix plot
cmd = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    final_pred, 
    display_labels=["Normal", "Attack"],
    cmap='Blues'
)
plt.title("Final IDS Confusion Matrix")
plt.savefig('final_confusion_matrix.png') # Saves the picture
plt.show()

# 8. CSV GENERATION (Requirement: CSV Output)
# Saves the results to a CSV for your final report
df_out = pd.DataFrame({
    "Actual_Label": y_test,
    "Stage_A_Predicted": clf_pred,
    "Final_Predicted": final_pred
})

df_out.to_csv("final_output_with_labels.csv", index=False)
print("\nSuccess: 'final_output_with_labels.csv' and 'final_confusion_matrix.png' generated.")
