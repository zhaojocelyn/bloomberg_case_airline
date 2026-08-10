# U.S. Airline Arrival Delay Prediction

A leakage-safe machine learning case study for predicting whether a U.S. domestic flight will arrive at least **15 minutes late**, using only information available at the flight's **scheduled departure time**.

The project combines Bureau of Transportation Statistics (BTS) flight data with hourly airport weather observations and builds an XGBoost risk model designed for operational use, temporal validation, and interpretable model diagnostics.

## Business question

> Can a flight be identified as likely to arrive at least 15 minutes late using only information available at scheduled departure?

The current answer is **yes**. The leading model, **Experiment 3b**, materially outperforms a prevalence-only baseline and provides useful ranking of higher-risk flights. It is best treated as a **risk-screening model** rather than a rule that automatically classifies every flight using a 0.50 probability threshold.

Please find detailed report here: https://github.com/zhaojocelyn/bloomberg_case_airline/blob/main/docs/airline_delay_case_study_report.pdf

## Project structure

The project uses Jupytext Python notebooks (`py:percent` format), so notebook cells appear as `# %%` blocks inside `.py` files.

```text
.
├── README.md              # summary + reproduction instructions
├── docs/                  # per-notebook specs
│   ├── airline_delay_case_study_report.pdf        
├── requirements.txt
├── setup.sh
├── jupytext.toml
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_modeling_stage1_*.ipynb
│   ├── 03_modeling_stage1_03b_history_recent_pressure.ipynb
│   └── 04_final_test_2026_experiment_3b.ipynb
├── src/
│   ├── download_bts.py    # loops the PREZIP endpoint (done)
│   ├── download_weather.py
│   ├── weather.py         # hourly table + rolling history
│   ├── clean_data.py      # target construction + split loader
│   ├── features.py        # holiday + weather features (done, tested)
│   ├── historical_features.py    
│   └── recent_performance_features.py    
├── data/
│   ├── raw/               # gitignored
│   ├── interim/           # parquet, gitignored (29 files present)
│   ├── lookups/           # small committed reference files
│   └── sample/            # one month, COMMITTED (dev only)
├── models/
│   ├── stage1_xgb_2024_to_2025_*.json               
│   └── stage1_xgb_2024_to_2025_*.joblib 
└── figures/
```

Key modules:

- `airline_delay_case_study_report.pdf ` — final report.
- `clean_data.py` — target construction, diversion recovery, model eligibility, and temporal split loading.
- `features.py` — schedule, calendar, holiday, timezone, weather, and scheduled-demand features.
- `historical_features.py` — leakage-safe static historical risk priors.
- `recent_performance_features.py` — lagged recent national, carrier, origin, and destination performance features.
- `03_modeling_stage1_03b_history_recent_pressure.py` — leading Stage 1 experiment, validation, threshold diagnostics, temporal stability, and TreeSHAP.
- `04_final_test_2026_experiment_3b.py` — retraining/evaluation workflow for the Jan-May 2026 out-of-time period.

## Running the project

### 1. Create an environment

A typical environment needs Python 3.11+ and the packages used throughout the project:

```bash
pip install pandas numpy pyarrow matplotlib scikit-learn xgboost \
    holidays timezonefinder joblib jupytext requests
```

### 2. Acquire and prepare the data

Run the acquisition workflow first:

```bash
jupytext --to notebook 01_data_acquisition.py
```

or open the `.py` file directly in a Jupytext-enabled Jupyter environment.

The modeling utilities expect monthly BTS parquet files under:

```text
data/interim/bts_YYYY_MM.parquet
```

### 3. Run exploratory analysis

```bash
jupytext --to notebook 02_eda.py
```

The EDA covers outcome prevalence, carriers, airports, time-of-day patterns, weather, demand interactions, anomalies, and recent operating regimes.

### 4. Run the leading Stage 1 experiment

```bash
jupytext --to notebook 03_modeling_stage1_03b_history_recent_pressure.py
```

This workflow builds Experiment 3b, evaluates it on 2025, produces threshold and monthly diagnostics, and computes TreeSHAP model-driver importance.

### 5. Run the out-of-time evaluation

```bash
jupytext --to notebook 04_final_test_2026_experiment_3b.py
```

The final-test workflow retrains the frozen specification using 2024-2025 development data and scores Jan-May 2026 without using 2026 for early stopping.

## Reproducibility notes

Target-derived features require special care:

- historical features for training rows are built only from earlier calendar months;
- later-period historical mappings are fitted on completed earlier development data;
- recent operating rates are computed chronologically with a 2-day availability lag;
- recent rates can update over time in an online-style backtest as sufficiently old outcomes become available;
- future weather observations are explicitly rejected;
- 2026 should not be used to select new features, hyperparameters, or thresholds after its results have been inspected.

## Limitations

- Airline delays are affected by extreme events and operational shocks that may not be predictable from scheduled-departure information.
- Carrier, airport, route, weather, historical-rate, and SHAP relationships are predictive rather than causal.
- The current Experiment 3b specification has not undergone the full planned hyperparameter search.
- 2025 served as both EDA and development/validation data, so repeated investigation may gradually adapt decisions to that year.
- The Jan-May 2026 period has already been examined; a later out-of-time period is preferable for unbiased evaluation of future model changes.
- The current target excludes cancellations and unrecovered diversions from ordinary arrival-delay classification.
