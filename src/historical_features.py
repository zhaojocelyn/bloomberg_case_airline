"""Leakage-safe, target-derived historical risk features.

How often has this carrier / airport / route been late before?
return: hist_*_delay_rate and hist_*_log_count

Empirical-Bayes shrinkage:
rate = (group_positives + smoothing × global_rate) / (group_count + smoothing)
log_count = log(1 + group_count)

The rate says how risky and the log_count says how confident we are in that estimate.
The smoothing constant acts as a number of imaginary prior flights pulling the estimate 
toward the global rate. 

Training rows receive expanding monthly estimates built only from earlier
calendar months. Validation or test rows receive mappings fitted on the full
preceding development period. For January 2024, the initial prior is set to 20%.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

HistoricalRateSpec: TypeAlias = tuple[str, tuple[str, ...], float]
HistoricalRateState: TypeAlias = dict[str, object]

__all__ = [
    "HistoricalRateSpec",
    "HistoricalRateState",
    "DEFAULT_HISTORICAL_RATE_SPECS",
    "historical_feature_names",
    "add_expanding_historical_rates",
    "fit_historical_rate_state",
    "apply_historical_rate_state",
]

# Smoothing is expressed as an equivalent number of prior flights.
# These are reasonable starting values, not fixed truths
DEFAULT_HISTORICAL_RATE_SPECS: tuple[HistoricalRateSpec, ...] = (
    ("carrier", ("Reporting_Airline",), 2_000.0),
    ("origin", ("Origin",), 1_000.0),
    ("dest", ("Dest",), 1_000.0),
    ("route", ("Origin", "Dest"), 200.0),
    ("carrier_route", ("Reporting_Airline", "Origin", "Dest"), 100.0),
    ("origin_hour", ("Origin", "sched_dep_hour"), 250.0),
)


def historical_feature_names(
    specs: tuple[HistoricalRateSpec, ...] = DEFAULT_HISTORICAL_RATE_SPECS,
) -> list[str]:
    """Return the generated model-feature names for ``specs``."""
    names: list[str] = []
    for name, _, _ in specs:
        names.extend([
            f"hist_{name}_delay_rate",
            f"hist_{name}_log_count",
        ])
    return names


def _validate_inputs(
    frame: pd.DataFrame,
    specs: tuple[HistoricalRateSpec, ...],
    *,
    date_col: str,
    target_col: str | None,
) -> None:
    required = {date_col}
    if target_col is not None:
        required.add(target_col)
    for _, columns, smoothing in specs:
        required.update(columns)
        if smoothing <= 0:
            raise ValueError("historical-rate smoothing must be positive")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing historical-feature columns: {missing}")


def _month_key(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum()):,} rows have invalid dates")
    return dates.dt.to_period("M").dt.to_timestamp()


def add_expanding_historical_rates(
    frame: pd.DataFrame,
    specs: tuple[HistoricalRateSpec, ...] = DEFAULT_HISTORICAL_RATE_SPECS,
    *,
    date_col: str = "FlightDate",
    target_col: str = "ArrDel15",
    initial_prior: float = 0.20,
) -> pd.DataFrame:
    """Add past-only historical rates to training rows.

    Every row in a calendar month uses outcomes from strictly earlier months.
    The current row, current day, and current month are never included. Groups
    with no prior support shrink fully to the past-only global delay rate.
    """
    _validate_inputs(
        frame,
        specs,
        date_col=date_col,
        target_col=target_col,
    )
    if not 0 < initial_prior < 1:
        raise ValueError("initial_prior must lie strictly between 0 and 1")

    out = frame.copy()
    out["__hist_row_order__"] = np.arange(len(out), dtype="int64")
    out["__hist_period__"] = _month_key(out[date_col])
    target = pd.to_numeric(out[target_col], errors="coerce")
    if target.isna().any() or ~target.isin([0, 1]).all():
        raise ValueError(f"{target_col!r} must contain only non-null 0/1 values")
    out["__hist_target__"] = target.astype("int8")

    global_monthly = (
        out.groupby("__hist_period__", observed=True)["__hist_target__"]
        .agg(positives="sum", count="size")
        .sort_index()
    )
    global_monthly["past_positives"] = (
        global_monthly["positives"].cumsum() - global_monthly["positives"]
    )
    global_monthly["past_count"] = (
        global_monthly["count"].cumsum() - global_monthly["count"]
    )
    global_monthly["__hist_global_rate__"] = (
        global_monthly["past_positives"] / global_monthly["past_count"]
    ).fillna(initial_prior)

    out = out.merge(
        global_monthly[["__hist_global_rate__"]],
        left_on="__hist_period__",
        right_index=True,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    for name, columns, smoothing in specs:
        rate_col = f"hist_{name}_delay_rate"
        count_col = f"hist_{name}_log_count"
        collisions = [c for c in (rate_col, count_col) if c in frame.columns]
        if collisions:
            raise ValueError(f"Historical output columns already exist: {collisions}")

        group_cols = [*columns, "__hist_period__"]
        monthly = (
            out.groupby(group_cols, observed=True, dropna=False)["__hist_target__"]
            .agg(positives="sum", count="size")
            .reset_index()
            .sort_values([*columns, "__hist_period__"])
        )
        grouped = monthly.groupby(
            list(columns),
            observed=True,
            dropna=False,
            sort=False,
        )
        monthly["__hist_group_positives__"] = (
            grouped["positives"].cumsum() - monthly["positives"]
        )
        monthly["__hist_group_count__"] = (
            grouped["count"].cumsum() - monthly["count"]
        )

        out = out.merge(
            monthly[
                group_cols
                + ["__hist_group_positives__", "__hist_group_count__"]
            ],
            on=group_cols,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        group_count = out.pop("__hist_group_count__").fillna(0)
        group_positives = out.pop("__hist_group_positives__").fillna(0)
        out[rate_col] = (
            (group_positives + smoothing * out["__hist_global_rate__"])
            / (group_count + smoothing)
        ).astype("float32")
        out[count_col] = np.log1p(group_count).astype("float32")

    out = (
        out.sort_values("__hist_row_order__")
        .drop(
            columns=[
                "__hist_row_order__",
                "__hist_period__",
                "__hist_target__",
                "__hist_global_rate__",
            ]
        )
        .reset_index(drop=True)
    )
    return out


def fit_historical_rate_state(
    frame: pd.DataFrame,
    specs: tuple[HistoricalRateSpec, ...] = DEFAULT_HISTORICAL_RATE_SPECS,
    *,
    target_col: str = "ArrDel15",
) -> HistoricalRateState:
    """Fit smoothed historical-rate mappings on a completed training period."""
    _validate_inputs(
        frame,
        specs,
        date_col="FlightDate",
        target_col=target_col,
    )
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.isna().any() or ~target.isin([0, 1]).all():
        raise ValueError(f"{target_col!r} must contain only non-null 0/1 values")

    work = frame.copy()
    work["__hist_target__"] = target.astype("int8")
    tables: dict[str, pd.DataFrame] = {}
    for name, columns, _ in specs:
        tables[name] = (
            work.groupby(list(columns), observed=True, dropna=False)["__hist_target__"]
            .agg(__hist_group_positives__="sum", __hist_group_count__="size")
            .reset_index()
        )

    return {
        "global_rate": float(target.mean()),
        "specs": specs,
        "tables": tables,
    }


def apply_historical_rate_state(
    frame: pd.DataFrame,
    state: HistoricalRateState,
) -> pd.DataFrame:
    """Apply frozen training-period historical mappings to later rows."""
    specs = state["specs"]
    tables = state["tables"]
    global_rate = float(state["global_rate"])
    if not isinstance(specs, tuple) or not isinstance(tables, dict):
        raise TypeError("Invalid historical-rate state")

    _validate_inputs(frame, specs, date_col="FlightDate", target_col=None)
    out = frame.copy()
    out["__hist_row_order__"] = np.arange(len(out), dtype="int64")

    for name, columns, smoothing in specs:
        rate_col = f"hist_{name}_delay_rate"
        count_col = f"hist_{name}_log_count"
        table = tables[name]
        out = out.merge(
            table,
            on=list(columns),
            how="left",
            validate="many_to_one",
            sort=False,
        )
        group_count = out.pop("__hist_group_count__").fillna(0)
        group_positives = out.pop("__hist_group_positives__").fillna(0)
        out[rate_col] = (
            (group_positives + smoothing * global_rate)
            / (group_count + smoothing)
        ).astype("float32")
        out[count_col] = np.log1p(group_count).astype("float32")

    return (
        out.sort_values("__hist_row_order__")
        .drop(columns="__hist_row_order__")
        .reset_index(drop=True)
    )
