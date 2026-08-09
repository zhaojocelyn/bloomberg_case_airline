"""Leakage-safe, target-derived recent operating-performance features.

How is the system running lately?
return: national_delay_rate_[7, 28]d and *_delay_rate_28d

The generated rates use only completed *earlier calendar days*. There is a 
two-day availability lag: a flight on day D draws on days D−2 back to D−29.
This models what an operational system would actually know at prediction time.

For temporal backtesting, ``add_recent_rates_to_splits`` combines ordered train
and validation partitions, computes past-only daily aggregates, and then restores
the original partitions. Validation rows may therefore use outcomes from earlier
validation days, matching an online system that is updated after outcomes become
available. The current day and the configured availability-lag days are always
excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd

__all__ = [
    "RecentGroupRateSpec",
    "DEFAULT_RECENT_GROUP_SPECS",
    "DEFAULT_NATIONAL_WINDOWS",
    "recent_feature_names",
    "add_past_only_recent_rates",
    "add_recent_rates_to_splits",
]

GroupColumns: TypeAlias = tuple[str, ...]


@dataclass(frozen=True)
class RecentGroupRateSpec:
    """Definition of one smoothed group-specific rolling delay rate."""

    name: str
    columns: GroupColumns
    window_days: int = 28
    smoothing: float = 250.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("RecentGroupRateSpec.name must not be empty")
        if not self.columns:
            raise ValueError("RecentGroupRateSpec.columns must not be empty")
        if self.window_days < 1:
            raise ValueError("window_days must be at least 1")
        if self.smoothing <= 0:
            raise ValueError("smoothing must be positive")


DEFAULT_NATIONAL_WINDOWS: tuple[int, ...] = (7, 28)
DEFAULT_RECENT_GROUP_SPECS: tuple[RecentGroupRateSpec, ...] = (
    RecentGroupRateSpec(
        name="carrier",
        columns=("Reporting_Airline",),
        window_days=28,
        smoothing=500.0,
    ),
    RecentGroupRateSpec(
        name="origin",
        columns=("Origin",),
        window_days=28,
        smoothing=250.0,
    ),
    RecentGroupRateSpec(
        name="dest",
        columns=("Dest",),
        window_days=28,
        smoothing=250.0,
    ),
)


def recent_feature_names(
    national_windows: tuple[int, ...] = DEFAULT_NATIONAL_WINDOWS,
    group_specs: tuple[RecentGroupRateSpec, ...] = DEFAULT_RECENT_GROUP_SPECS,
) -> list[str]:
    """Return the generated model-feature names."""
    names = [f"national_delay_rate_{window}d" for window in national_windows]
    names.extend(
        f"{spec.name}_delay_rate_{spec.window_days}d"
        for spec in group_specs
    )
    return names


def _validate_inputs(
    frame: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    national_windows: tuple[int, ...],
    group_specs: tuple[RecentGroupRateSpec, ...],
    initial_prior: float,
    availability_lag_days: int,
) -> None:
    if not 0 < initial_prior < 1:
        raise ValueError("initial_prior must lie strictly between 0 and 1")
    if availability_lag_days < 1:
        raise ValueError("availability_lag_days must be at least 1")
    if not national_windows or any(window < 1 for window in national_windows):
        raise ValueError("national_windows must contain positive integers")
    if len(set(national_windows)) != len(national_windows):
        raise ValueError("national_windows contains duplicates")

    required = {date_col, target_col}
    for spec in group_specs:
        required.update(spec.columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing recent-performance columns: {missing}")

    collisions = sorted(set(recent_feature_names(national_windows, group_specs)) & set(frame.columns))
    if collisions:
        raise ValueError(f"Recent-performance output columns already exist: {collisions}")

    target = pd.to_numeric(frame[target_col], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"{target_col!r} must contain only non-null 0/1 values")

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{int(dates.isna().sum()):,} rows have invalid dates")

    for spec in group_specs:
        missing_group_values = int(frame[list(spec.columns)].isna().any(axis=1).sum())
        if missing_group_values:
            raise ValueError(
                f"{missing_group_values:,} rows have missing values for "
                f"recent-rate group {spec.name!r}"
            )


def _rolling_past_sum(
    values: pd.Series,
    *,
    window_days: int,
    availability_lag_days: int,
) -> pd.Series:
    """Sum a dense daily series over past available days only."""
    return (
        values.shift(availability_lag_days)
        .rolling(window_days, min_periods=1)
        .sum()
    )


def _national_daily_table(
    work: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    full_dates: pd.DatetimeIndex,
    national_windows: tuple[int, ...],
    initial_prior: float,
    availability_lag_days: int,
) -> pd.DataFrame:
    daily = (
        work.groupby(date_col, observed=True)[target_col]
        .agg(__positives__="sum", __count__="size")
        .reindex(full_dates, fill_value=0)
        .rename_axis(date_col)
        .reset_index()
    )

    for window in national_windows:
        positives = _rolling_past_sum(
            daily["__positives__"],
            window_days=window,
            availability_lag_days=availability_lag_days,
        )
        count = _rolling_past_sum(
            daily["__count__"],
            window_days=window,
            availability_lag_days=availability_lag_days,
        )
        rate = positives.div(count.where(count.gt(0))).fillna(initial_prior)
        daily[f"national_delay_rate_{window}d"] = rate.astype("float32")

    return daily[[date_col, *recent_feature_names(national_windows, ())]]


def _group_daily_rate_table(
    work: pd.DataFrame,
    national_daily: pd.DataFrame,
    spec: RecentGroupRateSpec,
    *,
    date_col: str,
    target_col: str,
    full_dates: pd.DatetimeIndex,
    initial_prior: float,
    availability_lag_days: int,
) -> pd.DataFrame:
    group_cols = list(spec.columns)
    observed_daily = (
        work.groupby([*group_cols, date_col], observed=True, dropna=False)[target_col]
        .agg(__positives__="sum", __count__="size")
        .reset_index()
    )

    groups = observed_daily[group_cols].drop_duplicates().reset_index(drop=True)
    calendar = pd.DataFrame({date_col: full_dates})
    dense = groups.merge(calendar, how="cross")
    dense = dense.merge(
        observed_daily,
        on=[*group_cols, date_col],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    dense[["__positives__", "__count__"]] = dense[
        ["__positives__", "__count__"]
    ].fillna(0)
    dense = dense.sort_values([*group_cols, date_col]).reset_index(drop=True)

    grouped = dense.groupby(group_cols, observed=True, dropna=False, sort=False)
    past_positives = grouped["__positives__"].transform(
        lambda values: _rolling_past_sum(
            values,
            window_days=spec.window_days,
            availability_lag_days=availability_lag_days,
        )
    )
    past_count = grouped["__count__"].transform(
        lambda values: _rolling_past_sum(
            values,
            window_days=spec.window_days,
            availability_lag_days=availability_lag_days,
        )
    )

    national_reference_col = f"national_delay_rate_{spec.window_days}d"
    if national_reference_col not in national_daily.columns:
        raise KeyError(
            f"Group rate {spec.name!r} requires national window "
            f"{spec.window_days}d"
        )
    dense = dense.merge(
        national_daily[[date_col, national_reference_col]],
        on=date_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    prior = dense[national_reference_col].fillna(initial_prior)
    rate_col = f"{spec.name}_delay_rate_{spec.window_days}d"
    dense[rate_col] = (
        (past_positives.fillna(0) + spec.smoothing * prior)
        / (past_count.fillna(0) + spec.smoothing)
    ).astype("float32")

    return dense[[*group_cols, date_col, rate_col]]


def add_past_only_recent_rates(
    frame: pd.DataFrame,
    *,
    date_col: str = "FlightDate",
    target_col: str = "ArrDel15",
    national_windows: tuple[int, ...] = DEFAULT_NATIONAL_WINDOWS,
    group_specs: tuple[RecentGroupRateSpec, ...] = DEFAULT_RECENT_GROUP_SPECS,
    initial_prior: float = 0.20,
    availability_lag_days: int = 2,
) -> pd.DataFrame:
    """Add rolling delay rates using only earlier available calendar days.

    For a row on date ``D`` and an availability lag of two days, the newest
    outcome date that may contribute is ``D - 2``. A two-day default is
    conservative for a national network whose ``FlightDate`` values are local
    dates across multiple time zones. Set the lag to one only when the data
    pipeline guarantees that all previous-day outcomes are available at scoring
    time.

    Group-specific rates are smoothed toward the matching national rolling rate.
    Cold-start dates with no available history use ``initial_prior``.
    """
    _validate_inputs(
        frame,
        date_col=date_col,
        target_col=target_col,
        national_windows=national_windows,
        group_specs=group_specs,
        initial_prior=initial_prior,
        availability_lag_days=availability_lag_days,
    )

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise").dt.normalize()
    out["__recent_row_order__"] = np.arange(len(out), dtype="int64")
    out["__recent_target__"] = pd.to_numeric(
        out[target_col], errors="raise"
    ).astype("int8")

    full_dates = pd.date_range(out[date_col].min(), out[date_col].max(), freq="D")
    national_daily = _national_daily_table(
        out,
        date_col=date_col,
        target_col="__recent_target__",
        full_dates=full_dates,
        national_windows=national_windows,
        initial_prior=initial_prior,
        availability_lag_days=availability_lag_days,
    )
    out = out.merge(
        national_daily,
        on=date_col,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    for spec in group_specs:
        group_daily = _group_daily_rate_table(
            out,
            national_daily,
            spec,
            date_col=date_col,
            target_col="__recent_target__",
            full_dates=full_dates,
            initial_prior=initial_prior,
            availability_lag_days=availability_lag_days,
        )
        out = out.merge(
            group_daily,
            on=[*spec.columns, date_col],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    feature_cols = recent_feature_names(national_windows, group_specs)
    if out[feature_cols].isna().any().any():
        missing = out[feature_cols].isna().sum()
        raise AssertionError(
            "Recent-performance feature construction produced missing values: "
            f"{missing[missing.gt(0)].to_dict()}"
        )
    if ((out[feature_cols] < 0) | (out[feature_cols] > 1)).any().any():
        raise AssertionError("Recent delay-rate features must remain in [0, 1]")

    return (
        out.sort_values("__recent_row_order__")
        .drop(columns=["__recent_row_order__", "__recent_target__"])
        .reset_index(drop=True)
    )


def add_recent_rates_to_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    date_col: str = "FlightDate",
    target_col: str = "ArrDel15",
    national_windows: tuple[int, ...] = DEFAULT_NATIONAL_WINDOWS,
    group_specs: tuple[RecentGroupRateSpec, ...] = DEFAULT_RECENT_GROUP_SPECS,
    initial_prior: float = 0.20,
    availability_lag_days: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add online-updated recent rates to ordered train/validation splits.

    Validation features use training outcomes and earlier validation-day outcomes,
    subject to ``availability_lag_days``. This is appropriate for temporal
    backtesting of an operational model that updates after each completed day.
    It is not equivalent to a completely frozen validation feature mapping.
    """
    train_dates = pd.to_datetime(train[date_col], errors="coerce")
    valid_dates = pd.to_datetime(valid[date_col], errors="coerce")
    if train_dates.isna().any() or valid_dates.isna().any():
        raise ValueError("Train or validation split contains invalid dates")
    if train_dates.max().normalize() >= valid_dates.min().normalize():
        raise ValueError(
            "Train and validation date ranges must be strictly ordered and non-overlapping."
        )

    train_tagged = train.copy()
    valid_tagged = valid.copy()
    train_tagged["__recent_split__"] = "train"
    valid_tagged["__recent_split__"] = "valid"
    train_tagged["__recent_split_order__"] = np.arange(len(train), dtype="int64")
    valid_tagged["__recent_split_order__"] = np.arange(len(valid), dtype="int64")

    combined = pd.concat([train_tagged, valid_tagged], ignore_index=True, sort=False)
    combined = add_past_only_recent_rates(
        combined,
        date_col=date_col,
        target_col=target_col,
        national_windows=national_windows,
        group_specs=group_specs,
        initial_prior=initial_prior,
        availability_lag_days=availability_lag_days,
    )

    def restore(split: str) -> pd.DataFrame:
        return (
            combined.loc[combined["__recent_split__"].eq(split)]
            .sort_values("__recent_split_order__")
            .drop(columns=["__recent_split__", "__recent_split_order__"])
            .reset_index(drop=True)
        )

    train_out = restore("train")
    valid_out = restore("valid")
    if len(train_out) != len(train) or len(valid_out) != len(valid):
        raise AssertionError("Recent-rate split restoration changed row counts")
    return train_out, valid_out
