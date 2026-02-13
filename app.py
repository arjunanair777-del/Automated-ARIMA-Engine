from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from auto_arima_engine import AutoARIMAEngine


st.set_page_config(page_title="Automated ARIMA Engine", layout="wide")


def _build_demo_series(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    e = rng.normal(0.0, 1.0, n)
    x = np.zeros(n, dtype=float)
    for t in range(2, n):
        x[t] = 0.65 * x[t - 1] - 0.30 * x[t - 2] + e[t] + 0.45 * e[t - 1]
    return pd.DataFrame({"value": x})


def _prepare_series(df: pd.DataFrame, value_col: str, index_col: str | None) -> pd.Series:
    series = pd.to_numeric(df[value_col], errors="coerce")
    valid = series.notna()
    series = series[valid].astype(float)

    if index_col is not None:
        idx = pd.to_datetime(df.loc[valid, index_col], errors="coerce")
        keep = idx.notna()
        series = series[keep]
        idx = idx[keep]
        series.index = pd.DatetimeIndex(idx)
        series = series.sort_index()

    series = series.dropna()
    if series.empty:
        raise ValueError("No usable numeric observations found after cleaning.")
    if series.shape[0] < 30:
        raise ValueError("At least 30 observations are recommended for stable ARIMA estimation.")
    return series


def _series_with_index(series: pd.Series, values: np.ndarray) -> pd.Series:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return pd.Series(dtype=float)
    if v.size == series.shape[0]:
        return pd.Series(v, index=series.index, name=series.name)
    tail_index = series.index[-v.size :]
    return pd.Series(v, index=tail_index, name=series.name)


def _to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue().encode("utf-8")


def _run_engine(
    series: pd.Series,
    p_max: int,
    d_max: int,
    q_max: int,
    criterion: str,
    max_order_sum: int | None,
    top_n_diagnostics: int,
    lb_lag: int,
    alpha: float,
    n_jobs: int,
    forecast_steps: int,
) -> dict[str, Any]:
    engine = AutoARIMAEngine(
        p_max=p_max,
        d_max=d_max,
        q_max=q_max,
        criterion=criterion,
        max_order_sum=max_order_sum,
        top_n_diagnostics=top_n_diagnostics,
        lb_lag=lb_lag,
        alpha=alpha,
        n_jobs=n_jobs,
    )
    report = engine.fit(series)
    forecast = engine.forecast(steps=forecast_steps, alpha=alpha)

    fitted = _series_with_index(series, np.asarray(engine.best_result_.fittedvalues))
    residuals = _series_with_index(series, np.asarray(engine.best_result_.resid)).dropna()

    stationarity = report.stationarity_table.copy()
    comparison = report.comparison_table.copy()
    params = report.parameters.rename("estimate").to_frame()
    diagnostics = report.diagnostics.copy()
    summary_text = str(engine.best_result_.summary())

    return {
        "report_meta": {
            "best_order": report.best_order,
            "criterion": report.criterion,
            "best_score": float(report.best_score),
            "warnings": report.warnings,
        },
        "series": series,
        "fitted": fitted,
        "residuals": residuals,
        "forecast": forecast,
        "stationarity": stationarity,
        "comparison": comparison,
        "parameters": params,
        "diagnostics": diagnostics,
        "summary_text": summary_text,
    }


st.title("Automated ARIMA Engine")
st.caption(
    "Upload a time series and automatically select AR, MA, ARMA, or ARIMA "
    "using AIC/BIC with Box-Jenkins diagnostics."
)

uploaded_file = st.file_uploader("Input CSV", type=["csv"])
use_demo_data = st.checkbox("Use demo time series", value=uploaded_file is None)

if uploaded_file is None and not use_demo_data:
    st.info("Upload a CSV or enable demo data to continue.")
    st.stop()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = _build_demo_series()

if df.empty:
    st.error("Input file has no rows.")
    st.stop()

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in the input.")
    st.stop()

left, right = st.columns([2, 1])

with left:
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

with right:
    st.subheader("Configuration")
    value_col = st.selectbox("Value Column", options=numeric_cols, index=0)
    index_options = ["None"] + all_cols
    index_choice = st.selectbox("Index/Date Column (optional)", options=index_options, index=0)
    index_col = None if index_choice == "None" else index_choice

    criterion = st.selectbox("Selection Criterion", options=["bic", "aic"], index=0)
    p_max = st.slider("p_max", min_value=0, max_value=12, value=5)
    d_max = st.slider("d_max", min_value=0, max_value=3, value=2)
    q_max = st.slider("q_max", min_value=0, max_value=12, value=5)

    max_order_enabled = st.checkbox("Use p + q constraint", value=True)
    max_order_sum = st.slider("Max p + q", min_value=0, max_value=20, value=8) if max_order_enabled else None

    top_n_diagnostics = st.slider("Top N models for diagnostic filter", min_value=1, max_value=200, value=20)
    lb_lag = st.slider("Ljung-Box lag", min_value=1, max_value=50, value=10)
    alpha = st.number_input("Alpha (tests and forecast interval)", min_value=0.001, max_value=0.2, value=0.05, step=0.001, format="%.3f")
    n_jobs = st.slider("Parallel workers", min_value=1, max_value=16, value=1)
    forecast_steps = st.slider("Forecast steps", min_value=1, max_value=120, value=12)

    run_clicked = st.button("Run Auto ARIMA", type="primary", use_container_width=True)

if run_clicked:
    try:
        series = _prepare_series(df=df, value_col=value_col, index_col=index_col)
        with st.spinner("Fitting candidate ARIMA models and running diagnostics..."):
            results = _run_engine(
                series=series,
                p_max=p_max,
                d_max=d_max,
                q_max=q_max,
                criterion=criterion,
                max_order_sum=max_order_sum,
                top_n_diagnostics=top_n_diagnostics,
                lb_lag=lb_lag,
                alpha=alpha,
                n_jobs=n_jobs,
                forecast_steps=forecast_steps,
            )
        st.session_state["results"] = results
        st.success("Model search completed.")
    except Exception as exc:
        st.exception(exc)

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
meta = results["report_meta"]

for msg in meta["warnings"]:
    st.warning(msg)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Best Order", str(meta["best_order"]))
metric2.metric(f"Best {meta['criterion'].upper()}", f"{meta['best_score']:.3f}")
metric3.metric("Ljung-Box p-value", f"{results['diagnostics']['ljung_box_pvalue']:.4f}")
metric4.metric("Diagnostics Pass", "Yes" if results["diagnostics"]["passes_diagnostics"] else "No")

tab1, tab2, tab3, tab4 = st.tabs(["Series & Forecast", "Diagnostics", "Model Comparison", "Model Summary"])

with tab1:
    st.subheader("Observed vs Fitted")
    observed = results["series"].rename("observed")
    fitted = results["fitted"].rename("fitted")
    observed_fitted = pd.concat([observed, fitted], axis=1)
    st.line_chart(observed_fitted, use_container_width=True)

    st.subheader("Forecast")
    st.dataframe(results["forecast"], use_container_width=True)
    st.line_chart(results["forecast"][["forecast", "lower", "upper"]], use_container_width=True)

with tab2:
    st.subheader("Stationarity Identification")
    st.dataframe(results["stationarity"], use_container_width=True)

    st.subheader("Residual Diagnostics")
    diag_df = pd.DataFrame(
        [
            {"metric": "ljung_box_pvalue", "value": results["diagnostics"]["ljung_box_pvalue"]},
            {"metric": "jarque_bera_pvalue", "value": results["diagnostics"]["jarque_bera_pvalue"]},
            {"metric": "is_stable", "value": results["diagnostics"]["is_stable"]},
            {"metric": "is_white_noise", "value": results["diagnostics"]["is_white_noise"]},
            {"metric": "is_normal", "value": results["diagnostics"]["is_normal"]},
            {"metric": "passes_diagnostics", "value": results["diagnostics"]["passes_diagnostics"]},
        ]
    )
    st.dataframe(diag_df, use_container_width=True)

    st.subheader("Residual Series")
    st.line_chart(results["residuals"].rename("residual"), use_container_width=True)

with tab3:
    st.subheader("Model Comparison Table")
    max_rows = st.slider("Rows to display", min_value=10, max_value=500, value=100, step=10)
    comp = results["comparison"].copy().head(max_rows)
    st.dataframe(comp, use_container_width=True)

    st.download_button(
        label="Download comparison table (CSV)",
        data=_to_csv_bytes(results["comparison"], index=False),
        file_name="model_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download forecast (CSV)",
        data=_to_csv_bytes(results["forecast"], index=True),
        file_name="forecast.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download parameters (JSON)",
        data=json.dumps(results["parameters"]["estimate"].to_dict(), indent=2),
        file_name="best_model_parameters.json",
        mime="application/json",
    )

with tab4:
    st.subheader("Estimated Parameters")
    st.dataframe(results["parameters"], use_container_width=True)
    st.subheader("Statsmodels Summary")
    st.text(results["summary_text"])
