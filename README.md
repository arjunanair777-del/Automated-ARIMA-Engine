# Automated ARIMA Engine App

Fully functional Box-Jenkins ARIMA application with:
- automatic model search over AR(p), MA(q), ARMA(p,q), ARIMA(p,d,q)
- stationarity identification via ADF + KPSS
- model ranking by AIC/BIC
- residual diagnostics (Ljung-Box, Jarque-Bera, root stability)
- forecast generation with confidence intervals
- UI for upload, configuration, diagnostics, and downloads

## Install

```bash
pip install -r requirements.txt
```

## Run The App (Streamlit)

```bash
streamlit run app.py
```

Windows shortcut:

```bash
run_app.bat
```

## App Input

- CSV time-series file
- value column selection
- optional index/date column
- ARIMA search limits: `p_max`, `d_max`, `q_max`
- optional complexity cap: `p + q <= k`
- criterion: `AIC` or `BIC`
- diagnostics settings (`Ljung-Box lag`, `alpha`, top-N diagnostic pool)
- forecast horizon

## App Output

- best order `(p, d, q)`
- parameter estimates
- full model comparison table
- stationarity check table
- diagnostics summary
- forecast table and plot
Downloadable artifacts:
- `model_comparison.csv`
- `forecast.csv`
- `best_model_parameters.json`

## Python API

```python
import pandas as pd
from auto_arima_engine import AutoARIMAEngine

series = pd.read_csv("your_data.csv")["value"]

engine = AutoARIMAEngine(
    p_max=5,
    d_max=2,
    q_max=5,
    criterion="bic",
    max_order_sum=8,
    top_n_diagnostics=10,
    n_jobs=1,
)

report = engine.fit(series)
forecast = engine.forecast(steps=12)
```

## CLI Usage

```bash
python auto_arima_engine.py path/to/series.csv --column value --criterion bic --forecast-steps 12
```

## Top 10 Diseases Dashboard

Generate an Excel workbook (`top10_diseases_dashboard.xlsx`) with data on the
top 10 diseases in the world (WHO Global Health Estimates), including:

- **Data sheet** – formatted table with rank, disease name, category, annual
  deaths, prevalence, trend, and primary region
- **Dashboard sheet** – KPI summary cards, bar chart (annual deaths), pie chart
  (share of deaths), and a category breakdown table

### How to download

**Option 1 – Web UI (recommended):**  
Run the Streamlit app and click the **⬇️ Download Excel Dashboard** button in
the sidebar:

```bash
streamlit run app.py
```

**Option 2 – Command line:**

```bash
python top10_diseases_dashboard.py                   # saves to current directory
python top10_diseases_dashboard.py custom_name.xlsx   # custom output path
```

**Option 3 – Python API:**

```python
from top10_diseases_dashboard import generate_dashboard, generate_dashboard_bytes

generate_dashboard("my_report.xlsx")   # save to file
xlsx_bytes = generate_dashboard_bytes() # get raw bytes (e.g. for a web response)
```
