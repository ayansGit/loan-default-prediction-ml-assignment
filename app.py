import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide"
)

st.title("Loan Default Prediction")
st.markdown("Upload test data, select a model, and see predictions with evaluation metrics.")
st.divider()

# Model name → file mapping
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree":       "model/decision_tree.pkl",
    "KNN":                 "model/knn.pkl",
    "Naive Bayes":         "model/naive_bayes.pkl",
    "Random Forest":       "model/random_forest.pkl"
}

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Load metrics summary
metrics_df = pd.read_csv("model/metrics_summary.csv")

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:

    # Load uploaded CSV
    df_upload = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df_upload.head(10))

    # Separate features and target
    if "default" not in df_upload.columns:
        st.error("CSV must have a 'default' column as the target!")
    else:
        X_upload = df_upload.drop(columns=["default"])
        y_upload = df_upload["default"]

        # Scale using the saved scaler
        X_scaled = scaler.transform(X_upload)

        # Load the selected model
        model = joblib.load(MODEL_FILES[selected_model])

        # Make predictions
        y_pred      = model.predict(X_scaled)
        y_pred_prob = model.predict_proba(X_scaled)[:, 1]

        st.success(f"Predictions made using **{selected_model}**!")
        # ── Evaluation Metrics ──
        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy",  f"{accuracy_score(y_upload, y_pred):.4f}")
        col1.metric("Precision", f"{precision_score(y_upload, y_pred):.4f}")
        col2.metric("Recall",    f"{recall_score(y_upload, y_pred):.4f}")
        col2.metric("F1 Score",  f"{f1_score(y_upload, y_pred):.4f}")
        col3.metric("AUC Score", f"{roc_auc_score(y_upload, y_pred_prob):.4f}")
        col3.metric("MCC",       f"{matthews_corrcoef(y_upload, y_pred):.4f}")

        # ── Confusion Matrix ──
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_upload, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {selected_model}")
        st.pyplot(fig)

        # ── Model Comparison Table ──
        st.subheader("All Models Comparison")
        st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Please upload a CSV file from the sidebar to get started.")
