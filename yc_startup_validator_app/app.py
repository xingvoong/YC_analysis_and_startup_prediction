import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="YC Startup Validator",
    page_icon="🚀",
    layout="centered"
)

# -----------------------------
# Load model artifacts
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

vectorizer_path = MODEL_DIR / "vectorizer.pkl"
model_path = MODEL_DIR / "success_model.pkl"

model_loaded = True

try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

except FileNotFoundError:
    model_loaded = False

# -----------------------------
# Header
# -----------------------------
st.title("🚀 YC Startup Idea Validator")

st.write(
    """
    This tool evaluates a startup idea based on **patterns learned from historical YC companies**.

    It is **not a guarantee**, but a signal about *how similar your idea is to startups that scaled*.
    """
)

if model_loaded:
    st.success("ML model loaded successfully")
else:
    st.error(
        "Model files not found.\n\n"
        "Expected:\n"
        "- model/vectorizer.pkl\n"
        "- model/success_model.pkl"
    )

st.divider()

# -----------------------------
# User input
# -----------------------------
st.subheader("Describe your startup")

description = st.text_area(
    "What does your startup do?",
    height=150,
    placeholder="We build an AI agent that automates customer support for healthcare clinics..."
)

submitted = st.button("Evaluate Idea")

# -----------------------------
# Feature builder (TEXT ONLY)
# -----------------------------
def build_features(text: str):
    return vectorizer.transform([text.lower()])

# -----------------------------
# Helper: score interpretation
# -----------------------------
def interpret_score(score):
    if score >= 75:
        return "Very strong YC-style signal", "success"
    elif score >= 55:
        return "Promising, but highly competitive", "info"
    elif score >= 35:
        return "Weak signal — niche or execution-heavy", "warning"
    else:
        return "Low historical success similarity", "error"

# -----------------------------
# Helper: explain prediction (linear model)
# -----------------------------
def explain_prediction(X, top_k=8):
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    x_dense = X.toarray()[0]
    contributions = x_dense * coefs

    top_positive = contributions.argsort()[-top_k:][::-1]
    top_negative = contributions.argsort()[:top_k]

    return (
        feature_names[top_positive],
        contributions[top_positive],
        feature_names[top_negative],
        contributions[top_negative],
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    if not description.strip():
        st.warning("Please describe your startup idea.")
    elif not model_loaded:
        st.error("Model not loaded — cannot make prediction.")
    else:
        X = build_features(description)

        prob = model.predict_proba(X)[0][1]
        score = int(prob * 100)

        label, level = interpret_score(score)

        st.subheader("📊 Evaluation Result")
        st.metric("Predicted YC-Style Success Probability", f"{score}%")

        if level == "success":
            st.success(label)
        elif level == "info":
            st.info(label)
        elif level == "warning":
            st.warning(label)
        else:
            st.error(label)

        st.divider()

        # -----------------------------
        # Explanation section
        # -----------------------------
        st.subheader("🧠 Why the model thinks this")

        pos_feats, pos_vals, neg_feats, neg_vals = explain_prediction(X)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👍 Positive signals")
            for f, v in zip(pos_feats, pos_vals):
                if v > 0:
                    st.write(f"• **{f}**")

        with col2:
            st.markdown("### ⚠️ Negative signals")
            for f, v in zip(neg_feats, neg_vals):
                if v < 0:
                    st.write(f"• **{f}**")

        st.divider()

        # -----------------------------
        # Human-readable summary
        # -----------------------------
        st.subheader("📌 How to interpret this")

        st.write(
            f"""
            • This idea scores **{score}%**, meaning it resembles startups that scaled in YC
            • The model is influenced by **keywords and phrases** in your description
            • High scores favor:
                - Clear B2B value
                - Infrastructure, AI, or developer tooling language
                - Execution-focused descriptions

            Low scores often mean:
                - Consumer-only positioning
                - Vague problem statements
                - Crowded or undifferentiated markets
            """
        )

        st.caption(
            """
            Tip: Rewrite your description to be clearer about *who pays*, *what pain you solve*,
            and *why this is 10x better* — then re-run the evaluation.
            """
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Built from YC public data • Educational & exploratory use only"
)
