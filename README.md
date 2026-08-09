# U.S. Airline Arrival Delay Prediction

A leakage-safe machine learning case study for predicting whether a U.S. domestic flight will arrive at least **15 minutes late**, using only information available at the flight's **scheduled departure time**.

The project combines Bureau of Transportation Statistics (BTS) flight data with hourly airport weather observations and builds an XGBoost risk model designed for operational use, temporal validation, and interpretable model diagnostics.

## Business question

> Can a flight be identified as likely to arrive at least 15 minutes late using only information available at scheduled departure?

The current answer is **yes**. The leading model, **Experiment 3b**, materially outperforms a prevalence-only baseline and provides useful ranking of higher-risk flights. It is best treated as a **risk-screening model** rather than a rule that automatically classifies every flight using a 0.50 probability threshold.

## Prediction target

The Stage 1 target is:

```text
ArrDel15 = 1  if arrival delay >= 15 minutes
ArrDel15 = 0  otherwise
```

Cancelled flights and diversions that do not reach the scheduled destination are treated as separate non-arrival disruptions rather than ordinary non-delayed flights. For diversions that do reach the destination, arrival-delay targets are recovered from `DivArrDelay` when available.

## Data design

The project uses a temporal split:

| Period | Role |
|---|---|
| **2024** | Initial model training and historical feature construction |
| **2025** | Development, validation, EDA, feature comparison, threshold analysis |
| **Jan-May 2026** | Out-of-time evaluation / demonstration check |

The 2026 period has now been inspected, so it should not be reused as a newly untouched test set after additional model changes. Any future tuning should be evaluated on a later out-of-time period.

## Leakage policy

Every model feature must be available at the scheduled-departure prediction point.

Post-departure or outcome-time variables are excluded, including:

- actual departure delay;
- actual departure time;
- taxi-out time;
- actual elapsed time;
- cancellation/diversion outcomes;
- BTS delay-cause fields such as carrier, weather, NAS, security, and late-aircraft delay minutes.

Additional safeguards include:

- historical target-derived features use only prior periods;
- recent operating rates use a **2-day availability lag**;
- origin and destination weather are aligned to the same physical scheduled-departure instant;
- weather timestamp-age checks prevent future observations from entering a row;
- categorical levels are learned from development data, with unseen later values handled explicitly.

## Feature groups

Experiment 3b combines the following feature families.

### Schedule and calendar
Carrier, origin, destination, month, day of week, day of month, scheduled departure hour/minute, scheduled elapsed time, distance, and holiday proximity/name.

### Weather
Origin weather includes visibility, ceiling, flight category, wind, precipitation, frozen precipitation, thunderstorms, temperature, recent IFR exposure, recent precipitation, and recent gusts. Destination weather is observed at the scheduled-departure instant and uses a smaller leakage-safe subset.

### Scheduled airport demand
Scheduled origin departures/operations and destination arrivals within the relevant airport-hour.

### Static historical priors
Smoothed historical delay rates and support counts for carrier, origin, destination, route, carrier-route, and origin-departure-hour.

### Recent operating conditions
Leakage-safe rolling delay rates include national 7-day and 28-day rates plus 28-day carrier, origin, and destination rates.

### National delay pressure
Experiment 3b adds:

```text
national_delay_pressure_7d_vs_28d
    = national_delay_rate_7d
    - national_delay_rate_28d
```

Positive values indicate that recent network conditions are worse than the broader 28-day regime; negative values indicate improvement.

## Modeling experiments

All four feature experiments used the same XGBoost configuration so that changes in validation performance could be attributed primarily to the feature specification.

| Model | Main addition | PR-AUC | ROC-AUC | Log loss | Brier | ECE-10 | Top-10% delay capture |
|---|---|---:|---:|---:|---:|---:|---:|
| No-skill baseline | 2025 prevalence | 0.2250 | 0.5000 | 0.5339 | 0.1746 | 0.0151 | 0.0994 |
| Experiment 1 | Base schedule/weather/demand | 0.4176 | 0.6982 | 0.4880 | 0.1581 | 0.0170 | 0.2212 |
| Experiment 2 | + historical priors | 0.4235 | 0.7020 | 0.4877 | 0.1578 | 0.0259 | 0.2244 |
| Experiment 3 | + recent operating rates | 0.4197 | 0.7015 | 0.4863 | 0.1576 | 0.0120 | 0.2234 |
| **Experiment 3b** | **+ national pressure** | **0.4250** | **0.7032** | **0.4847** | **0.1570** | **0.0096** | **0.2255** |

Experiment 3b is the current leading feature specification. Hyperparameter tuning was intentionally deferred because of time constraints, so it should be viewed as the **leading candidate rather than a fully optimized final model**.

## Why these metrics?

- **PR-AUC** measures ranking quality for the minority delay class while emphasizing precision and recall.
- **ROC-AUC** measures how well the model separates delayed from non-delayed flights across thresholds.
- **Log loss** evaluates probability quality and heavily penalizes confident incorrect predictions.
- **Brier score** measures squared error between predicted probabilities and observed binary outcomes.
- **ECE-10** summarizes calibration error across 10 probability bins.
- **Top-10% delay capture** measures the share of all delayed flights found among the 10% highest-scored flights.

## Operating thresholds and alert capacity

The diagnostic max-F1 threshold on 2025 validation is approximately **0.217**, but this is not automatically the best operational threshold.

A business may instead define an alert capacity and act on only the highest-risk fraction of flights. This keeps operational workload fixed and makes model comparisons easier.

For the Jan-May 2026 out-of-time check:

| Alert fraction | Flights flagged | Score cutoff | Precision | Delay capture |
|---:|---:|---:|---:|---:|
| 5% | 140,691 | 0.4905 | 57.9% | 13.5% |
| 10% | 281,381 | 0.4100 | 49.6% | 23.1% |
| 20% | 562,762 | 0.3264 | 41.8% | 38.8% |
| 30% | 844,142 | 0.2728 | 37.1% | 51.7% |

This illustrates the operational trade-off: smaller alert groups provide higher precision, while larger alert groups capture more of the total delay population.

## Temporal stability

Monthly evaluation is important because airline operations are non-stationary.

The Jan-May 2026 check showed:

- ranking skill remained positive across all five months;
- monthly PR-AUC lift was roughly **1.9x the monthly no-skill baseline**;
- monthly Brier skill remained positive;
- aggregate calibration was reasonably close through January-April;
- May showed noticeable probability overprediction, while ranking performance remained useful.

This suggests that ranking is more stable than absolute probability calibration and that calibration should be monitored over time.

## Model interpretation

The project uses **TreeSHAP** on a reproducible sample of 2025 validation flights to explain the leading Experiment 3b model.

The highest global SHAP drivers include:

1. `sched_dep_hour`
2. `hist_origin_hour_delay_rate`
3. `hist_carrier_route_delay_rate`
4. `Dest`
5. `national_delay_rate_7d`
6. `Origin`
7. `carrier_delay_rate_28d`
8. `DayOfWeek`
9. `dest_flight_category`
10. `days_to_next_holiday`

Scheduled departure hour is the strongest global model driver in the current SHAP analysis.

SHAP importance answers **which features most influence model predictions**. Signed SHAP values can additionally show whether particular feature values push a prediction toward higher or lower delay risk.

SHAP and XGBoost gain importance are not causal measures. They explain model behavior and predictive associations, not what would happen if an airline or airport actively changed a feature.

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

## Next step: Stage 2 delay severity

Stage 1 answers:

> How likely is this flight to arrive at least 15 minutes late?

A future Stage 2 regression model could answer:

> If the flight is significantly delayed, how severe is that delay likely to be?

Stage 2 would use `ArrDelayMinutes` as the conditional target among flights with actual arrival delays of at least 15 minutes while preserving the same scheduled-departure information boundary.

Rather than collapsing probability and severity into a single operational number, the two outputs can support a simple probability-severity matrix:

| | Low severity | High severity |
|---|---|---|
| **Low probability** | Monitor / low priority | Tail risk |
| **High probability** | Routine friction | Act now |

This preserves the distinction between common manageable delays and lower-probability high-impact disruptions and provides a more interpretable framework for operational prioritization.

## Current conclusion

Experiment 3b is the strongest current Stage 1 candidate. It combines schedule, calendar, weather, demand, historical performance, recent operating conditions, and a national delay-pressure feature while maintaining a strict scheduled-departure leakage boundary.

The model provides useful delay-risk ranking, reasonable probability quality, and interpretable operational targeting. The next development steps are disciplined hyperparameter tuning on a valid development period, continued calibration monitoring, directional SHAP analysis, and the proposed Stage 2 conditional-severity extension.
