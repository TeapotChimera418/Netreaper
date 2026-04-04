import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from stage_a_baseline import preprocess_data, train_model

np.random.seed(42)

st.set_page_config(layout="wide")

st.title("🚀 NetReaper - Adversarial IDS Demo")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("KDDTrain_with_headers.csv")

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Controls")

use_attack = st.sidebar.checkbox("Enable Adversarial Attack", value=False)

noise_level = st.sidebar.slider(
    "Attack Strength",
    min_value=0.0,
    max_value=1.9,
    value=1.3,
    step=0.05
)

# =========================
# PREPROCESS + MODEL
# =========================
X_train, X_test, y_train, y_test = preprocess_data(df)

clf_model = train_model(X_train, y_train, "rf")

# =========================
# STAGE A
# =========================
clf_pred = clf_model.predict(X_test)
acc_stage_a = accuracy_score(y_test, clf_pred)

# =========================
# STAGE B (OPTIONAL ATTACK)
# =========================
if use_attack:
    # Strong + consistent attack
    noise = np.random.normal(0, noise_level, X_test.shape)
    X_input = X_test + noise

    clf_pred_input = clf_model.predict(X_input)
    acc_adv = accuracy_score(y_test, clf_pred_input)
else:
    X_input = X_test
    clf_pred_input = clf_pred
    acc_adv = None

# =========================
# ADVERSARIAL ATTACK BLOCK
# =========================

if use_attack:
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, X_test.shape)
    X_adv = X_test + noise

    # Predict on attacked data
    clf_pred_adv = clf_model.predict(X_adv)

    acc_adv = (y_test == clf_pred_adv).mean()

    st.metric("After Attack Accuracy", f"{acc_adv:.4f}")

else:
    st.metric("After Attack Accuracy", "Disabled")
    
# =========================
# STAGE C
# =========================
iso_model = joblib.load("stage_c_isolation_forest.pkl")

X_test_df = pd.DataFrame(X_input)

if hasattr(iso_model, "feature_names_in_"):
    missing_cols = set(iso_model.feature_names_in_) - set(X_test_df.columns)
    for col in missing_cols:
        X_test_df[col] = 0
    X_test_df = X_test_df[iso_model.feature_names_in_]

anomaly_pred = iso_model.predict(X_test_df)

# =========================
# FINAL DECISION
# =========================
final_pred = clf_pred_input.copy()
final_pred[anomaly_pred == -1] = 1

acc_final = accuracy_score(y_test, final_pred)

# =========================
# METRICS (AUTO-DETECT COLUMNS)
# =========================

st.subheader("Model Performance")

df = pd.read_csv("final_output_with_labels.csv")

st.write("Detected columns:", df.columns.tolist())

# Try matching possible column names
if "Actual_Label" in df.columns:
    y_true = df["Actual_Label"]
elif "Actual" in df.columns:
    y_true = df["Actual"]
else:
    st.error("No Actual column found!")
    st.stop()

if "Stage_A_Predicted" in df.columns:
    y_stage_a = df["Stage_A_Predicted"]
elif "Predicted" in df.columns:
    y_stage_a = df["Predicted"]
else:
    y_stage_a = None

if "Final_Predicted" in df.columns:
    y_final = df["Final_Predicted"]
elif "Predicted" in df.columns:
    y_final = df["Predicted"]
else:
    st.error("No Final prediction column found!")
    st.stop()

# Compute accuracies
if y_stage_a is not None:
    acc_stage_a = (y_true == y_stage_a).mean()
    st.metric("Stage A (Baseline)", f"{acc_stage_a:.4f}")

acc_final = (y_true == y_final).mean()
st.metric("Final System", f"{acc_final:.4f}")

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("📉 Confusion Matrix (Final)")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=ax)
st.pyplot(fig)

# =========================
# DISTRIBUTION
# =========================
st.subheader("📊 Prediction Distribution")

dist_df = pd.DataFrame({
    "Stage A": pd.Series(clf_pred).value_counts(),
    "After Attack": pd.Series(clf_pred_input).value_counts(),
    "Final": pd.Series(final_pred).value_counts()
}).fillna(0)

st.bar_chart(dist_df)

# =========================
# SAMPLE OUTPUT
# =========================
st.subheader("🔍 Sample Predictions")

sample_df = pd.DataFrame({
    "Actual": y_test[:50].values,
    "Stage A": clf_pred[:50],
    "After Attack": clf_pred_input[:50],
    "Final": final_pred[:50]
})

st.dataframe(sample_df)

# =========================
# EXPLANATION
# =========================
st.subheader("🧠 Explanation")

st.write(f"""
- Noise Level: **{noise_level}**
- Stage A detects known attacks
- Attack degrades model performance
- Isolation Forest catches anomalies
- Final system improves robustness
""")
