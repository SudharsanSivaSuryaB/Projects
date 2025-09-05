import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")

st.title("📈 Stock Time Series — Forecast Comparison")
st.markdown("Load prediction CSVs from the pipeline (located in `artifacts/`) and compare models.")

artifacts_dir = Path("artifacts")
files = sorted(artifacts_dir.glob("predictions_*.csv"))
if not files:
    st.info("No prediction files found. Run `python run_pipeline.py` first.")
else:
    model_files = {f.stem.replace("predictions_",""): f for f in files}
    model = st.selectbox("Select model to view", list(model_files.keys()))
    df = pd.read_csv(model_files[model], parse_dates=["Date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["y_true"], name="Actual"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["y_pred"], name=f"Predicted — {model}"))
    fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metrics")
    comp_path = artifacts_dir / "comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info("Run pipeline to generate `comparison.csv`.")
