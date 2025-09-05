# Time Series Stock Market — End‑to‑End Project

This project implements **time series analysis and forecasting** for stock market data using **ARIMA, SARIMA, Prophet, and LSTM**, with clear **preprocessing**, **EDA**, **model evaluation**, and an optional **Streamlit** app.

> Built to satisfy the requirements in the provided PDFs: *Internship Project Collaboration Guidelines* and *Time series stock market* (objectives, tools, models, deliverables).

## Quick Start

```bash
# 1) Create a fresh environment (recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install dependencies (may take a while for Prophet/TensorFlow)
pip install -r requirements.txt

# 3) Run the pipeline (downloads data with yfinance)
python run_pipeline.py --ticker AAPL --start 2015-01-01 --end 2025-08-31 --horizon 30

# 4) (Optional) Launch the Streamlit app
streamlit run app/StreamlitApp.py
```

Artifacts (plots, metrics, predictions) will be saved under `artifacts/`.

## Project Structure

```
stock-timeseries-project/
├─ app/
│  └─ StreamlitApp.py
├─ artifacts/
│  ├─ figures/
│  └─ (metrics.csv, predictions_*.csv, comparison.csv, ...)
├─ notebooks/
│  └─ 01_quick_eda.ipynb
├─ src/
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ eda.py
│  ├─ evaluate.py
│  ├─ compare.py
│  └─ models/
│     ├─ arima_sarima.py
│     ├─ prophet_model.py
│     └─ lstm_model.py
├─ run_pipeline.py
├─ requirements.txt
└─ README.md
```

## Deliverables Covered

- ✅ Cleaned dataset & preprocessing code (`src/preprocess.py`).
- ✅ Models: ARIMA/SARIMA (`src/models/arima_sarima.py`), Prophet (`src/models/prophet_model.py`), LSTM (`src/models/lstm_model.py`).
- ✅ Evaluation & comparison (`src/evaluate.py`, `src/compare.py`).
- ✅ Visualizations (saved to `artifacts/figures/`).
- ✅ Optional web deployment (`app/StreamlitApp.py`).
- ✅ Reproducible pipeline (`run_pipeline.py`) and documentation (`README.md`).

## Notes

- **Prophet build**: Installs `cmdstanpy` which compiles Stan models on first run (downloads ~100MB). Internet and build tools are required.
- **TensorFlow**: Optional but included to meet requirements; CPU version installs by default.
- **No API keys**: Data is pulled from `yfinance` (free).

---

**Authoring Tip**: If running in restricted environments (no internet), place your own CSV with columns: `Date, Open, High, Low, Close, Volume` and use `--csv path/to/file.csv`.
