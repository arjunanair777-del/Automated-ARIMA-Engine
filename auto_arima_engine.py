"""Automated ARIMA engine using a Box-Jenkins style workflow.

Features:
- Identification: selects differencing order via ADF + KPSS.
- Estimation: fits ARIMA(p, d, q) candidates via MLE.
- Diagnostics: Ljung-Box, optional Jarque-Bera, root stability.
- Selection: picks best model by AIC/BIC after filtering failures.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class StationarityCheck:
    selected_d: int
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateResult:
    order: tuple[int, int, int]
    aic: float
    bic: float
    llf: float
    n_params: int
    lb_pvalue: float
    jb_pvalue: float
    is_stable: bool
    is_white_noise: bool
    is_normal: bool
    passes_diagnostics: bool
    score: float
    error: str | None = None
    model_result: Any | None = None


@dataclass
class AutoARIMAReport:
    best_order: tuple[int, int, int]
    criterion: str
    best_score: float
    stationarity_table: pd.DataFrame
    comparison_table: pd.DataFrame
    parameters: pd.Series
    diagnostics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _as_series(values: pd.Series | pd.DataFrame | Iterable[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one column.")
        series = values.iloc[:, 0].copy()
    else:
        series = pd.Series(values)

    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        raise ValueError("Input series is empty after removing non-numeric values.")
    return series.astype(float)


def _difference(series: pd.Series, d: int) -> pd.Series:
    if d <= 0:
        return series.dropna()
    diffed = series.copy()
    for _ in range(d):
        diffed = diffed.diff().dropna()
    return diffed


def _safe_adf_pvalue(series: pd.Series) -> float:
    try:
        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return np.nan


def _safe_kpss_pvalue(series: pd.Series) -> float:
    try:
        return float(kpss(series, regression="c", nlags="auto")[1])
    except Exception:
        return np.nan


def _root_condition(roots: np.ndarray) -> bool:
    if roots.size == 0:
        return True
    return bool(np.all(np.abs(roots) > 1.0))


class AutoARIMAEngine:
    def __init__(
        self,
        p_max: int = 5,
        d_max: int = 2,
        q_max: int = 5,
        criterion: str = "bic",
        max_order_sum: int | None = None,
        top_n_diagnostics: int = 10,
        lb_lag: int = 10,
        alpha: float = 0.05,
        n_jobs: int = 1,
    ) -> None:
        if criterion.lower() not in {"aic", "bic"}:
            raise ValueError("criterion must be either 'aic' or 'bic'.")
        if p_max < 0 or d_max < 0 or q_max < 0:
            raise ValueError("p_max, d_max, and q_max must be non-negative.")
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")

        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.criterion = criterion.lower()
        self.max_order_sum = max_order_sum
        self.top_n_diagnostics = top_n_diagnostics
        self.lb_lag = lb_lag
        self.alpha = alpha
        self.n_jobs = n_jobs

        self.best_result_: Any | None = None
        self.best_order_: tuple[int, int, int] | None = None
        self.comparison_table_: pd.DataFrame | None = None
        self.stationarity_table_: pd.DataFrame | None = None
        self.fit_warnings_: list[str] = []

    def _determine_d(self, series: pd.Series) -> StationarityCheck:
        records: list[dict[str, Any]] = []
        selected_d = self.d_max
        warnings_list: list[str] = []

        for d in range(self.d_max + 1):
            diffed = _difference(series, d)
            adf_p = _safe_adf_pvalue(diffed)
            kpss_p = _safe_kpss_pvalue(diffed)

            adf_stationary = bool(adf_p < self.alpha) if not np.isnan(adf_p) else False
            # KPSS null hypothesis is stationarity, so p > alpha supports stationarity.
            kpss_stationary = bool(kpss_p > self.alpha) if not np.isnan(kpss_p) else True
            stationary = adf_stationary and kpss_stationary

            records.append(
                {
                    "d": d,
                    "n_obs": int(diffed.shape[0]),
                    "adf_pvalue": adf_p,
                    "kpss_pvalue": kpss_p,
                    "adf_stationary": adf_stationary,
                    "kpss_stationary": kpss_stationary,
                    "selected": stationary,
                }
            )

            if stationary:
                selected_d = d
                break

        if not any(r["selected"] for r in records):
            warnings_list.append(
                f"Series did not pass stationarity checks up to d={self.d_max}; using d={self.d_max}."
            )

        table = pd.DataFrame(records)
        return StationarityCheck(selected_d=selected_d, table=table, warnings=warnings_list)

    def _candidate_orders(self, d: int) -> list[tuple[int, int, int]]:
        orders: list[tuple[int, int, int]] = []
        for p in range(self.p_max + 1):
            for q in range(self.q_max + 1):
                if self.max_order_sum is not None and (p + q > self.max_order_sum):
                    continue
                orders.append((p, d, q))
        return orders

    def _fit_candidate(self, series: pd.Series, order: tuple[int, int, int]) -> CandidateResult:
        p, d, q = order
        try:
            model = ARIMA(series, order=(p, d, q))
            fitted = model.fit()
            resid = pd.Series(np.asarray(fitted.resid)).dropna()

            lag = self.lb_lag
            if resid.shape[0] <= lag:
                lag = max(1, resid.shape[0] - 1)

            if lag >= 1 and resid.shape[0] > 2:
                lb_pvalue = float(
                    acorr_ljungbox(resid, lags=[lag], return_df=True)["lb_pvalue"].iloc[0]
                )
            else:
                lb_pvalue = np.nan

            if resid.shape[0] > 7:
                jb_pvalue = float(jarque_bera(resid).pvalue)
            else:
                jb_pvalue = np.nan

            ar_roots = np.asarray(getattr(fitted, "arroots", np.array([])))
            ma_roots = np.asarray(getattr(fitted, "maroots", np.array([])))
            is_stable = _root_condition(ar_roots) and _root_condition(ma_roots)
            is_white_noise = bool(lb_pvalue > self.alpha) if not np.isnan(lb_pvalue) else False
            is_normal = bool(jb_pvalue > self.alpha) if not np.isnan(jb_pvalue) else False
            passes = is_stable and is_white_noise

            aic = float(fitted.aic)
            bic = float(fitted.bic)
            score = aic if self.criterion == "aic" else bic

            return CandidateResult(
                order=order,
                aic=aic,
                bic=bic,
                llf=float(fitted.llf),
                n_params=int(len(fitted.params)),
                lb_pvalue=lb_pvalue,
                jb_pvalue=jb_pvalue,
                is_stable=is_stable,
                is_white_noise=is_white_noise,
                is_normal=is_normal,
                passes_diagnostics=passes,
                score=score,
                model_result=fitted,
            )
        except Exception as exc:
            return CandidateResult(
                order=order,
                aic=np.nan,
                bic=np.nan,
                llf=np.nan,
                n_params=0,
                lb_pvalue=np.nan,
                jb_pvalue=np.nan,
                is_stable=False,
                is_white_noise=False,
                is_normal=False,
                passes_diagnostics=False,
                score=np.inf,
                error=str(exc),
                model_result=None,
            )

    def _fit_all_candidates(
        self, series: pd.Series, orders: list[tuple[int, int, int]]
    ) -> list[CandidateResult]:
        if self.n_jobs == 1:
            return [self._fit_candidate(series, order) for order in orders]

        results: list[CandidateResult] = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(self._fit_candidate, series, order): order for order in orders}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def fit(self, series: pd.Series | pd.DataFrame | Iterable[float]) -> AutoARIMAReport:
        ts = _as_series(series)
        self.fit_warnings_ = []

        stationarity = self._determine_d(ts)
        self.fit_warnings_.extend(stationarity.warnings)
        selected_d = stationarity.selected_d

        orders = self._candidate_orders(selected_d)
        candidate_results = self._fit_all_candidates(ts, orders)

        successful = [r for r in candidate_results if r.error is None]
        if not successful:
            raise RuntimeError("All candidate models failed to fit.")

        successful_sorted = sorted(successful, key=lambda r: r.score)
        if self.top_n_diagnostics > 0:
            diagnostic_pool = successful_sorted[: self.top_n_diagnostics]
        else:
            diagnostic_pool = successful_sorted

        passing = [r for r in diagnostic_pool if r.passes_diagnostics]
        if not passing:
            self.fit_warnings_.append(
                "No model passed diagnostics in the selected pool; falling back to best score model."
            )
            best = successful_sorted[0]
        else:
            best = min(passing, key=lambda r: r.score)

        self.best_result_ = best.model_result
        self.best_order_ = best.order
        self.stationarity_table_ = stationarity.table

        diagnostic_orders = {r.order for r in diagnostic_pool}

        rows: list[dict[str, Any]] = []
        for r in candidate_results:
            p, d, q = r.order
            rows.append(
                {
                    "order": str(r.order),
                    "p": p,
                    "d": d,
                    "q": q,
                    "aic": r.aic,
                    "bic": r.bic,
                    "llf": r.llf,
                    "n_params": r.n_params,
                    "lb_pvalue": r.lb_pvalue,
                    "jb_pvalue": r.jb_pvalue,
                    "is_stable": r.is_stable,
                    "is_white_noise": r.is_white_noise,
                    "is_normal": r.is_normal,
                    "passes_diagnostics": r.passes_diagnostics,
                    "selected_pool": r.order in diagnostic_orders,
                    "score": r.score,
                    "error": r.error,
                }
            )

        sort_cols = [self.criterion] + [c for c in ("aic", "bic") if c != self.criterion]
        comparison = pd.DataFrame(rows).sort_values(by=sort_cols, ascending=True, na_position="last")
        self.comparison_table_ = comparison

        params = pd.Series(self.best_result_.params)  # type: ignore[union-attr]
        diagnostics = {
            "ljung_box_pvalue": best.lb_pvalue,
            "jarque_bera_pvalue": best.jb_pvalue,
            "is_stable": best.is_stable,
            "is_white_noise": best.is_white_noise,
            "is_normal": best.is_normal,
            "passes_diagnostics": best.passes_diagnostics,
        }

        return AutoARIMAReport(
            best_order=best.order,
            criterion=self.criterion,
            best_score=best.score,
            stationarity_table=stationarity.table,
            comparison_table=comparison,
            parameters=params,
            diagnostics=diagnostics,
            warnings=self.fit_warnings_.copy(),
        )

    def forecast(self, steps: int = 12, alpha: float = 0.05) -> pd.DataFrame:
        if self.best_result_ is None:
            raise RuntimeError("Call fit() before forecast().")
        if steps <= 0:
            raise ValueError("steps must be > 0.")

        pred = self.best_result_.get_forecast(steps=steps)
        frame = pred.summary_frame(alpha=alpha).rename(
            columns={
                "mean": "forecast",
                "mean_ci_lower": "lower",
                "mean_ci_upper": "upper",
            }
        )
        return frame


def auto_arima_engine(
    series: pd.Series | pd.DataFrame | Iterable[float],
    p_max: int = 5,
    d_max: int = 2,
    q_max: int = 5,
    criterion: str = "bic",
    max_order_sum: int | None = None,
    top_n_diagnostics: int = 10,
    lb_lag: int = 10,
    alpha: float = 0.05,
    n_jobs: int = 1,
    forecast_steps: int = 0,
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
    output: dict[str, Any] = {
        "best_order": report.best_order,
        "criterion": report.criterion,
        "score": report.best_score,
        "parameters": report.parameters,
        "diagnostics": report.diagnostics,
        "stationarity_table": report.stationarity_table,
        "comparison_table": report.comparison_table,
        "warnings": report.warnings,
        "model_summary": engine.best_result_.summary() if engine.best_result_ is not None else None,
    }
    if forecast_steps > 0:
        output["forecast"] = engine.forecast(forecast_steps)
    return output


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Automated ARIMA Engine")
    parser.add_argument("csv_path", type=Path, help="Path to input CSV file")
    parser.add_argument("--column", type=str, default=None, help="Target numeric column name")
    parser.add_argument("--index-column", type=str, default=None, help="Optional index/date column")
    parser.add_argument("--p-max", type=int, default=5)
    parser.add_argument("--d-max", type=int, default=2)
    parser.add_argument("--q-max", type=int, default=5)
    parser.add_argument("--criterion", choices=["aic", "bic"], default="bic")
    parser.add_argument("--max-order-sum", type=int, default=None)
    parser.add_argument("--top-n-diagnostics", type=int, default=10)
    parser.add_argument("--lb-lag", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--forecast-steps", type=int, default=12)
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=Path("model_comparison.csv"),
        help="CSV output path for model comparison table",
    )
    parser.add_argument(
        "--forecast-out",
        type=Path,
        default=Path("forecast.csv"),
        help="CSV output path for forecasts",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    if args.column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found. Provide --column.")
        column = numeric_cols[0]
    else:
        column = args.column
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in input file.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if args.index_column is not None:
        if args.index_column not in df.columns:
            raise ValueError(f"Index column '{args.index_column}' not found in input file.")
        series.index = pd.to_datetime(df.loc[series.index, args.index_column], errors="coerce")
        series = series[~series.index.isna()]

    engine = AutoARIMAEngine(
        p_max=args.p_max,
        d_max=args.d_max,
        q_max=args.q_max,
        criterion=args.criterion,
        max_order_sum=args.max_order_sum,
        top_n_diagnostics=args.top_n_diagnostics,
        lb_lag=args.lb_lag,
        alpha=args.alpha,
        n_jobs=args.n_jobs,
    )
    report = engine.fit(series)
    forecast = engine.forecast(steps=args.forecast_steps)

    report.comparison_table.to_csv(args.comparison_out, index=False)
    forecast.to_csv(args.forecast_out, index=True)

    print(f"Best order: {report.best_order}")
    print(f"Best {report.criterion.upper()}: {report.best_score:.4f}")
    print("Diagnostics:", report.diagnostics)
    if report.warnings:
        print("Warnings:")
        for msg in report.warnings:
            print(f"- {msg}")
    print(f"Saved comparison table to: {args.comparison_out}")
    print(f"Saved forecast to: {args.forecast_out}")


if __name__ == "__main__":
    _cli()
