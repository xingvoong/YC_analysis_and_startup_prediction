"""
Component 4 — Drift monitoring dashboard
Run: streamlit run phase2/observability/dashboard.py
"""

import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "predictions.db"

st.set_page_config(page_title="YC Validator — Monitoring", layout="wide")
st.title("Prediction Monitoring")

if not DB_PATH.exists():
    st.warning("No predictions logged yet. Make some requests to /predict first.")
    st.stop()

df = pd.read_sql("SELECT * FROM predictions ORDER BY ts DESC", sqlite3.connect(DB_PATH))
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date

if df.empty:
    st.info("No data yet.")
    st.stop()

# ── Summary metrics ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total predictions", len(df))
col2.metric("Avg score", f"{df['score'].mean():.1f}")
col3.metric("% strong/promising", f"{(df['level'].isin(['strong','promising']).mean() * 100):.1f}%")
col4.metric("Model version", df["model_ver"].iloc[0])

st.divider()

# ── Score distribution ────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Score distribution")
    st.bar_chart(df["score"].value_counts().sort_index())

with col_b:
    st.subheader("Level breakdown")
    st.bar_chart(df["level"].value_counts())

st.divider()

# ── Score over time ───────────────────────────────────────────────────────────
st.subheader("Average score over time")
daily = df.groupby("date")["score"].mean().reset_index()
st.line_chart(daily.set_index("date"))

# ── Drift flag ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Drift check")

WINDOW = 50
if len(df) >= WINDOW * 2:
    baseline = df.iloc[-WINDOW * 2 : -WINDOW]["score"].mean()
    recent = df.iloc[-WINDOW:]["score"].mean()
    delta = abs(recent - baseline)
    if delta > 10:
        st.error(f"Drift detected — baseline avg {baseline:.1f} vs recent avg {recent:.1f} (delta {delta:.1f})")
    else:
        st.success(f"No drift — baseline avg {baseline:.1f} vs recent avg {recent:.1f} (delta {delta:.1f})")
else:
    st.info(f"Need at least {WINDOW * 2} predictions to run drift check. Have {len(df)} so far.")

st.divider()
st.subheader("Recent predictions")
st.dataframe(df.head(50)[["ts", "score", "level", "desc_len", "model_ver"]], use_container_width=True)
