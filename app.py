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

# Load Data
df = pd.read_csv("KDDTrain_with_headers.csv")

# Sidebar
st.sidebar.header("⚙️ Controls")

noise_level = st.sidebar.slider(
    "Attack Strength",
    min_value=0.0,
    max_value=1.9,
    value=1.3,
    step=0.05
)

# Preprocessor model
X_train, X_test, y_train, y_test = preprocess_data(df)

clf_model = train_model(X_train, y_train, "rf")

# Stage A
clf_pred = clf_model.predict(X_test)
acc_stage_a = accuracy_score(y_test, clf_pred)

#Stage B
X_adv = X_test * np.random.uniform(
    1 - noise_level,
    1 + noise_level,
    X_test.shape
)

clf_pred_adv = clf_model.predict(X_adv)
acc_adv = accuracy_score(y_test, clf_pred_adv)

# Stage C
iso_model = joblib.load("stage_c_isolation_forest.pkl")

X_test_df = pd.DataFrame(X_adv)

if hasattr(iso_model, "feature_names_in_"):
    missing_cols = set(iso_model.feature_names_in_) - set(X_test_df.columns)
    for col in missing_cols:
        X_test_df[col] = 0
    X_test_df = X_test_df[iso_model.feature_names_in_]

anomaly_pred = iso_model.predict(X_test_df)

# Final Decision
final_pred = clf_pred_adv.copy()
final_pred[anomaly_pred == -1] = 1

acc_final = accuracy_score(y_test, final_pred)

# Metrics display
st.subheader("📊 Accuracy Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Stage A (Baseline)", f"{acc_stage_a:.4f}")
col2.metric("After Attack", f"{acc_adv:.4f}")
col3.metric("Final System", f"{acc_final:.4f}")

# Confusion matrix
st.subheader("📉 Confusion Matrix (Final)")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=ax)
st.pyplot(fig)

#Distribution
st.subheader("📊 Prediction Distribution")

dist_df = pd.DataFrame({
    "Stage A": pd.Series(clf_pred).value_counts(),
    "After Attack": pd.Series(clf_pred_adv).value_counts(),
    "Final": pd.Series(final_pred).value_counts()
}).fillna(0)

st.bar_chart(dist_df)

# Sample input
st.subheader("🔍 Sample Predictions")

sample_df = pd.DataFrame({
    "Actual": y_test[:50].values,
    "Stage A": clf_pred[:50],
    "After Attack": clf_pred_adv[:50],
    "Final": final_pred[:50]
})

st.dataframe(sample_df)

#Explaination
st.subheader("🧠 Explanation")

st.write(f"""
- Noise Level: **{noise_level}**
- Stage A detects known attacks
- Attack degrades model performance
- Isolation Forest catches anomalies
- Final system improves robustness
""")
