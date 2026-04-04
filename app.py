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

@st.cache_data
def load_training_data() -> pd.DataFrame:
    return pd.read_csv("KDDTrain_with_headers.csv")


@st.cache_data
def load_final_output() -> pd.DataFrame:
    return pd.read_csv("final_output_with_labels.csv")


@st.cache_resource
def build_artifacts(df: pd.DataFrame):
    x_train, x_test, y_train, y_test = preprocess_data(df)
    model = train_model(x_train, y_train, "rf")
    return model, x_test, y_test


@st.cache_resource
def load_iso_model():
    return joblib.load("stage_c_isolation_forest.pkl")


df = load_training_data()
clf_model, X_test, y_test = build_artifacts(df)
iso_model = load_iso_model()

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
# STAGE C
# =========================
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

output_df = load_final_output()

st.write("Detected columns:", output_df.columns.tolist())

# Try matching possible column names
if "Actual_Label" in output_df.columns:
    y_true = output_df["Actual_Label"]
elif "Actual" in output_df.columns:
    y_true = output_df["Actual"]
else:
    st.error("No Actual column found!")
    st.stop()

if "Stage_A_Predicted" in output_df.columns:
    y_stage_a = output_df["Stage_A_Predicted"]
elif "Predicted" in output_df.columns:
    y_stage_a = output_df["Predicted"]
else:
    y_stage_a = None

if "Final_Predicted" in output_df.columns:
    y_final = output_df["Final_Predicted"]
elif "Predicted" in output_df.columns:
    y_final = output_df["Predicted"]
else:
    st.error("No Final prediction column found!")
    st.stop()

# Compute accuracies
if y_stage_a is not None:
    acc_stage_a_file = (y_true == y_stage_a).mean()
else:
    acc_stage_a_file = acc_stage_a

acc_final_file = (y_true == y_final).mean()

attack_delta = None if acc_adv is None else acc_adv - acc_stage_a
recovery_delta = acc_final - (acc_adv if acc_adv is not None else acc_stage_a)

tab_dashboard, tab_detailed, tab_raw = st.tabs([
    "Dashboard",
    "Detailed Analysis",
    "Raw Data",
])

with tab_dashboard:
    c1, c2, c3 = st.columns(3)

    c1.metric("Stage A (Baseline)", f"{acc_stage_a:.4f}")
    if acc_adv is None:
        c2.metric("After Attack Accuracy", "Disabled")
    else:
        c2.metric("After Attack Accuracy", f"{acc_adv:.4f}", delta=f"{attack_delta:+.4f}")
    c3.metric("Final System", f"{acc_final:.4f}", delta=f"{recovery_delta:+.4f}")

    c4, c5 = st.columns(2)
    c4.metric("Stage A (CSV)", f"{acc_stage_a_file:.4f}")
    c5.metric("Final (CSV)", f"{acc_final_file:.4f}")

    st.subheader("📊 Prediction Distribution")
    dist_df = pd.DataFrame({
        "Stage A": pd.Series(clf_pred).value_counts(),
        "After Attack": pd.Series(clf_pred_input).value_counts(),
        "Final": pd.Series(final_pred).value_counts(),
    }).fillna(0)
    st.bar_chart(dist_df)

with tab_detailed:
    st.subheader("📉 Confusion Matrices")

    fig_a, ax_a = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, clf_pred, ax=ax_a)
    ax_a.set_title("Stage A")
    st.pyplot(fig_a)

    if use_attack:
        fig_b, ax_b = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, clf_pred_input, ax=ax_b)
        ax_b.set_title("After Attack")
        st.pyplot(fig_b)

    fig_c, ax_c = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=ax_c)
    ax_c.set_title("Final")
    st.pyplot(fig_c)

with tab_raw:
    st.subheader("🔍 Sample Predictions")
    sample_df = pd.DataFrame({
        "Actual": y_test[:50].values,
        "Stage A": clf_pred[:50],
        "After Attack": clf_pred_input[:50],
        "Final": final_pred[:50],
    })
    st.dataframe(sample_df)

    st.subheader("🧾 Output CSV Snapshot")
    st.dataframe(output_df.head(100))

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
