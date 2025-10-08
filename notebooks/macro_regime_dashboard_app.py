# streamlit app: interactive macro regime dashboard
# Run: streamlit run notebooks/macro_regime_dashboard_app.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Macro Regime Dashboard",
    layout="wide",
    page_icon="📈"
)

DATA_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

DEFAULT_FILES = {
    "regimes": DATA_DIR / "regime_probabilities.csv",
    "ns_factors": DATA_DIR / "ns_factors.csv",
    "backtest": DATA_DIR / "backtest_results.csv",
    "weights": DATA_DIR / "portfolio_weights.csv",
    "label_map": DATA_DIR / "regime_labels.csv",          # optional (Regime_Index,Label)
    "var_forecast": DATA_DIR / "var_forecast.csv",        # optional
}

PLOT_COLORS = {
    "Regime_0": "#1f77b4",
    "Regime_1": "#ff7f0e",
    "Regime_2": "#2ca02c",
    "Level": "#1f77b4",
    "Slope": "#ff7f0e",
    "Curvature": "#2ca02c",
    "CumReturn": "#111111",
}

# -----------------------------
# Load helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path, index_col=0, parse_dates=True) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        df = df.sort_index()
        return df
    except Exception as e:
        return None

@st.cache_data(show_spinner=False)
def load_all(files: dict):
    regimes = load_csv(files["regimes"])
    ns = load_csv(files["ns_factors"])
    backtest = load_csv(files["backtest"])
    weights = load_csv(files["weights"])
    label_map = None
    if files["label_map"].exists():
        try:
            lm = pd.read_csv(files["label_map"])
            if {"Regime_Index", "Label"}.issubset(lm.columns):
                label_map = dict(zip(lm["Regime_Index"], lm["Label"]))
        except Exception:
            label_map = None
    var_fc = load_csv(files["var_forecast"])
    return regimes, ns, backtest, weights, label_map, var_fc

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Settings")

# File pickers (keep defaults but allow overrides)
paths = {}
for key, default in DEFAULT_FILES.items():
    paths[key] = Path(st.sidebar.text_input(f"{key} file", str(default)))

# Load data
regimes, ns, backtest, weights, label_map, var_fc = load_all(paths)

# Date range selector (use anything that exists)
date_min = None
date_max = None
for df in [regimes, ns, backtest]:
    if df is not None and len(df) > 0:
        d0, d1 = df.index.min(), df.index.max()
        date_min = d0 if date_min is None else min(date_min, d0)
        date_max = d1 if date_max is None else max(date_max, d1)

if date_min is None:
    st.error("No data found. Check file paths in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
sel_range = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date()
)
if isinstance(sel_range, tuple):
    start_date, end_date = pd.to_datetime(sel_range[0]), pd.to_datetime(sel_range[1])
else:
    start_date, end_date = date_min, date_max

st.sidebar.markdown("---")
show_overlay = st.sidebar.checkbox("Overlay regimes on portfolio", value=True)
smooth_probs = st.sidebar.checkbox("Smooth regime probabilities (EMA-3)", value=False)

# -----------------------------
# Header & data summary
# -----------------------------
st.title("📈 Macro Regime Dashboard")

def shape_or_dash(df):
    return "-" if df is None else str(df.shape)

st.caption(
    f"Loaded — "
    f"Regimes: {shape_or_dash(regimes)} | "
    f"NS factors: {shape_or_dash(ns)} | "
    f"Portfolio: {shape_or_dash(backtest)} | "
    f"Weights: {shape_or_dash(weights)}"
)

# -----------------------------
# Panel 1: Regime Probabilities
# -----------------------------
st.subheader("Regime Probabilities Over Time")

if regimes is None:
    st.info("No regime_probabilities.csv found.")
else:
    regs = regimes.copy()
    regs = regs.loc[(regs.index >= start_date) & (regs.index <= end_date)]

    regime_cols = [c for c in regs.columns if c.startswith("Regime_")]
    if smooth_probs and len(regime_cols) > 0:
        regs[regime_cols] = regs[regime_cols].ewm(span=3, adjust=False).mean()

    fig = go.Figure()
    for c in regime_cols:
        fig.add_trace(
            go.Scatter(
                x=regs.index, y=regs[c],
                name=c, mode="lines",
                line=dict(width=2, color=PLOT_COLORS.get(c, None))
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Probability", range=[-0.02, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Panel 2: NS Factors
# -----------------------------
st.subheader("Nelson–Siegel Yield Curve Factors")

if ns is None:
    st.info("No ns_factors.csv found.")
else:
    nsv = ns.copy()
    # normalize index to month end for alignment
    nsv.index = pd.to_datetime(nsv.index).to_period("M").to_timestamp("M")
    nsv = nsv.loc[(nsv.index >= start_date) & (nsv.index <= end_date)]

    fig = go.Figure()
    for c in ["Level", "Slope", "Curvature"]:
        if c in nsv.columns:
            fig.add_trace(
                go.Scatter(
                    x=nsv.index, y=nsv[c], name=c, mode="lines",
                    line=dict(width=2, color=PLOT_COLORS.get(c, None))
                )
            )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Factor value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Panel 3: Portfolio Performance & Drawdowns
# -----------------------------
st.subheader("Portfolio Performance and Drawdowns")

if backtest is None or "cumret" not in backtest.columns:
    st.info("No backtest_results.csv with 'cumret' found.")
else:
    bt = backtest.copy()
    bt = bt.loc[(bt.index >= start_date) & (bt.index <= end_date)]
    bt["peak"] = bt["cumret"].cummax()
    bt["drawdown"] = (bt["cumret"] / bt["peak"]) - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index, y=bt["cumret"],
            name="Cumulative Return",
            mode="lines",
            line=dict(width=2, color=PLOT_COLORS["CumReturn"])
        )
    )
    fig.add_trace(
        go.Bar(
            x=bt.index, y=bt["drawdown"],
            name="Drawdown",
            marker_color="rgba(200,0,0,0.25)",
            yaxis="y2"
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Cum Return"),
        yaxis2=dict(overlaying="y", side="right", title="Drawdown"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Panel 4: Portfolio with Regime Overlays
# -----------------------------
st.subheader("Portfolio Return with Regime Overlays")

if (backtest is not None) and (regimes is not None):
    # Align to monthly end for consistent overlays
    bt = backtest.copy()
    bt.index = pd.to_datetime(bt.index).to_period("M").to_timestamp("M")
    regs = regimes.copy()
    regs.index = pd.to_datetime(regs.index).to_period("M").to_timestamp("M")

    # Clip to range
    bt = bt.loc[(bt.index >= start_date) & (bt.index <= end_date)]
    regs = regs.loc[(regs.index >= start_date) & (regs.index <= end_date)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt.index, y=bt["cumret"],
            name="Cumulative Return",
            mode="lines",
            line=dict(color="#111111", width=2)
        )
    )

    if show_overlay:
        regime_cols = [c for c in regs.columns if c.startswith("Regime_")]
        # “Dominant regime” per month
        if regime_cols:
            dom = regs[regime_cols].idxmax(axis=1)

            for rcol in regime_cols:
                mask = dom == rcol
                # find contiguous blocks for this regime
                # create segments where mask changes
                blocks = []
                in_block = False
                start_idx = None
                for i, (dt, m) in enumerate(mask.items()):
                    if m and not in_block:
                        in_block = True
                        start_idx = dt
                    if in_block and (not m or i == len(mask)-1):
                        end_idx = dt if not m else dt
                        blocks.append((start_idx, end_idx))
                        in_block = False

                for (s, e) in blocks:
                    fig.add_vrect(
                        x0=s, x1=e,
                        fillcolor=PLOT_COLORS.get(rcol, "gray"),
                        opacity=0.09,
                        layer="below",
                        line_width=0,
                        annotation_text=rcol,
                        annotation_position="top left",
                        annotation=dict(font_size=10, bgcolor="rgba(255,255,255,0.5)")
                    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need both backtest_results.csv and regime_probabilities.csv to build the overlay.")

# -----------------------------
# Optional: latest VAR forecast
# -----------------------------
st.subheader("Latest VAR Forecast (optional)")

if var_fc is None:
    st.info("No var_forecast.csv found (optional).")
else:
    regime_cols = [c for c in var_fc.columns if c.startswith("Regime_")]
    last = var_fc.iloc[-1][regime_cols] if regime_cols else None
    if last is not None and len(last) > 0:
        st.write("Next-step regime probabilities (from VAR):")
        st.dataframe(last.to_frame("probability").style.format("{:.3f}"))
    else:
        st.info("var_forecast.csv has no Regime_* columns.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built from saved artifacts in data/processed — adjust paths in the sidebar if needed.")

