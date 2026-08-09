"""Revised feature engineering for the BTS flight-delay case study.

Filename: ``features_revised.py``. This version explicitly exports the airport
coordinate, timezone, destination-departure-key, scheduled-demand, holiday, and
weather helpers used by ``03_modeling_stage1_revised.py``.

Design note
-----------
Holiday features are a pure function of the calendar date. The analysis window
holds ~880 distinct dates against ~17M flights, so features are computed once on
a date-level calendar and then joined onto the flight table. Computing them
row-wise over 17M rows would be several orders of magnitude more work for an
identical result.

Leakage
-------
Holiday features are exempt from the train-only rule that governs aggregate
features (carrier historical on-time rate, route mean delay, target encodings).
The civil calendar is published years ahead, so a holiday flag for a test-period
date uses no information from the test period's outcomes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

__all__ = [
    "build_holiday_calendar",
    "add_holiday_features",
    "derive_flight_category",
    "add_weather_history",
    "scheduled_departure_timestamp",
    "load_airport_coordinates",
    "build_airport_timezone_lookup",
    "add_destination_departure_keys",
    "scheduled_departure_hour",
    "add_schedule_demand",
    "add_weather_features",
    "ORIGIN_WEATHER_COLS",
    "DEST_WEATHER_COLS",
]

# --------------------------------------------------------------------------- #
# Weather column contracts
# --------------------------------------------------------------------------- #

#: Carried from the origin airport. Everything observable at scheduled departure.
ORIGIN_WEATHER_COLS = [
    "visibility_sm",
    "ceiling_ft",
    "flight_category",
    "wind_speed_kt",
    "wind_gust_kt",
    "precip_mm",
    "frozen_precip",
    "thunderstorm",
    "temp_c",
    "ifr_hours_past3",
    "precip_3h",
    "max_gust_3h",
]

#: Carried from the destination airport, as observed at DEPARTURE time.
#: Deliberately a subset -- see add_weather_features for why arrival-time
#: destination weather is excluded entirely.
DEST_WEATHER_COLS = [
    "flight_category",
    "wind_gust_kt",
    "frozen_precip",
    "thunderstorm",
]

#: Ceiling/visibility thresholds for the standard aviation categories.
#: A category applies when EITHER the ceiling OR the visibility falls in range,
#: which is why this cannot be replaced by two independent binned features.
_IFR_OR_WORSE = ("IFR", "LIFR")


def build_holiday_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Return one row per calendar date between ``start`` and ``end``.

    Columns
    -------
    FlightDate               datetime64[ns]
    days_to_next_holiday     int16   0 on the holiday itself
    days_since_prev_holiday  int16   0 on the holiday itself
    next_holiday_name        category
    prev_holiday_name        category

    Notes
    -----
    Two directional distances are carried rather than one signed distance to the
    nearest holiday. A signed-nearest feature collapses whenever a date sits
    between two holidays: 27 December is two days after Christmas *and* five days
    before New Year, and only one of those survives. The Christmas-New Year
    corridor is the most disrupted stretch of the travel calendar, so that is
    precisely the case worth representing properly.

    The `holidays` package emits observed-shift entries (for example
    "Independence Day (observed)") alongside actual dates, so building distances
    from its output handles observed shifts automatically.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError(f"start {start.date()} is after end {end.date()}")

    # Pad by a year either side so dates near the window edge still find a
    # neighbouring holiday in both directions.
    years = range(start.year - 1, end.year + 2)
    us = holidays.US(years=years)

    hol_dates = pd.DatetimeIndex(sorted(us.keys()))
    hol_names = np.array([us[d.date()] for d in hol_dates], dtype=object)

    cal = pd.date_range(start, end, freq="D")

    # Integer day numbers make the neighbour search a simple searchsorted.
    cal_i = cal.values.astype("datetime64[D]").astype(np.int64)
    hol_i = hol_dates.values.astype("datetime64[D]").astype(np.int64)

    # searchsorted with side="left": index of the first holiday >= each date.
    nxt = np.searchsorted(hol_i, cal_i, side="left")
    # ...and the last holiday <= each date.
    prv = np.searchsorted(hol_i, cal_i, side="right") - 1

    # The one-year pad makes out-of-range indices impossible, but clip defensively
    # so a caller passing an odd range gets sane values rather than an IndexError.
    nxt_c = np.clip(nxt, 0, len(hol_i) - 1)
    prv_c = np.clip(prv, 0, len(hol_i) - 1)

    out = pd.DataFrame(
        {
            "FlightDate": cal,
            "days_to_next_holiday": (hol_i[nxt_c] - cal_i).astype("int16"),
            "days_since_prev_holiday": (cal_i - hol_i[prv_c]).astype("int16"),
            "next_holiday_name": pd.Categorical(hol_names[nxt_c]),
            "prev_holiday_name": pd.Categorical(hol_names[prv_c]),
        }
    )
    return out


def add_holiday_features(
    df: pd.DataFrame,
    date_col: str = "FlightDate",
) -> pd.DataFrame:
    """Join holiday features onto a flight-level table.

    The calendar is unique on date, so the merge is many-to-one by construction
    and cannot duplicate flight rows. This is asserted rather than assumed.
    """
    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} not in dataframe; columns: {list(df.columns)[:8]}...")

    # BTS parquet stores FlightDate as a string, so coerce rather than assume.
    # Normalising to midnight guards against any stray time component, which
    # would otherwise silently fail to match the calendar and yield all-null
    # features instead of an error.
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    if dates.isna().any():
        raise ValueError(f"{dates.isna().sum():,} rows have an unparseable {date_col}")

    cal = build_holiday_calendar(dates.min(), dates.max())

    n_before = len(df)
    key = "__holiday_join_key__"
    out = (
        df.assign(**{key: dates})
        .merge(
            cal.rename(columns={"FlightDate": key}),
            on=key,
            how="left",
            validate="many_to_one",
        )
        .drop(columns=key)
    )

    if len(out) != n_before:
        raise AssertionError(f"merge changed row count: {n_before:,} -> {len(out):,}")
    if out["days_to_next_holiday"].isna().any():
        raise AssertionError("holiday join left nulls; calendar range is too narrow")
    return out


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #


def derive_flight_category(
    ceiling_ft: pd.Series,
    visibility_sm: pd.Series,
) -> pd.Series:
    """Classify conditions into VFR, MVFR, IFR, or LIFR.

    The worst applicable category wins because ceiling and visibility are an
    OR rule. VFR is assigned only when both inputs are present and both meet VFR
    limits. Missing inputs therefore remain missing rather than being treated as
    favorable weather.
    """
    c = pd.to_numeric(ceiling_ft, errors="coerce")
    v = pd.to_numeric(visibility_sm, errors="coerce")

    cat = pd.Series(pd.NA, index=c.index, dtype="object")
    known = c.notna() & v.notna()

    cat.loc[known] = "VFR"
    cat.loc[known & ((c <= 3000) | (v <= 5))] = "MVFR"
    cat.loc[known & ((c < 1000) | (v < 3))] = "IFR"
    cat.loc[known & ((c < 500) | (v < 1))] = "LIFR"

    return pd.Categorical(
        cat,
        categories=["VFR", "MVFR", "IFR", "LIFR"],
        ordered=True,
    )


def add_weather_history(
    wx: pd.DataFrame,
    window: int = 3,
    airport_col: str = "airport",
    hour_col: str = "hour",
    include_current: bool = True,
) -> pd.DataFrame:
    """Add rolling weather-history features on a complete hourly grid.

    Missing-hour semantics
    ----------------------
    ``rolling(...).sum(min_periods=1)`` skips NaN rather than propagating it, so:

    - **Partially observed window** -> counts only the hours actually observed.
      ``ifr_hours_past3`` therefore reads as "hours *known* to be IFR or worse",
      not "hours that were IFR". A gap understates rather than invents.
    - **Fully unobserved window** -> ``<NA>``, since there is nothing to sum.

    Preserving NaN in the input is what makes the second case possible
    and stops a gap being reported as confirmed good weather.

    Missing flight categories stay missing; they are not counted as VFR.
    Repeated carried-forward METAR snapshots do not duplicate the same
    precipitation amount in ``precip_3h``.
    """
    if window < 1:
        raise ValueError("window must be at least 1")

    required = {airport_col, hour_col, "flight_category"}
    missing = sorted(required - set(wx.columns))
    if missing:
        raise KeyError(f"weather table is missing required columns: {missing}")

    out = wx.copy()
    out[hour_col] = pd.to_datetime(out[hour_col], errors="coerce")
    if out[hour_col].isna().any():
        raise ValueError(
            f"{out[hour_col].isna().sum():,} weather rows have an invalid "
            f"{hour_col!r}"
        )

    out = out.sort_values([airport_col, hour_col]).reset_index(drop=True)

    duplicated = out.duplicated([airport_col, hour_col])
    if duplicated.any():
        raise ValueError(
            f"weather table has {int(duplicated.sum()):,} duplicate "
            f"({airport_col}, {hour_col}) rows"
        )

    deltas = out.groupby(airport_col, observed=True)[hour_col].diff().dropna()
    bad_steps = deltas.ne(pd.Timedelta("1h"))
    if bad_steps.any():
        raise ValueError(
            f"weather series contains {int(bad_steps.sum()):,} non-hourly "
            "steps; reindex to a complete hourly grid before rolling"
        )

    airport_key = out[airport_col]
    category = out["flight_category"].astype("string")
    category_known = category.notna()

    # Preserve missing weather. A naive .isin(...).astype(int) would turn
    # missing categories into zero and incorrectly count them as non-IFR.
    is_ifr = pd.Series(np.nan, index=out.index, dtype="float32")
    is_ifr.loc[category_known] = (
        category.loc[category_known]
        .isin(_IFR_OR_WORSE)
        .astype("float32")
    )

    if not include_current:
        is_ifr = is_ifr.groupby(airport_key, observed=True).shift(1)

    out["ifr_hours_past3"] = (
        is_ifr.groupby(airport_key, observed=True)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .round()
        .astype("Int8")
    )

    if "precip_mm" in out.columns:
        precip_source = pd.to_numeric(out["precip_mm"], errors="coerce")

        # weather.py may carry one safe METAR snapshot into a later hour when a
        # routine report is missing. p01i belongs to the observation, so count
        # it once rather than once per carried-forward snapshot.
        if "observation_time_local" in out.columns:
            observation_time = pd.to_datetime(
                out["observation_time_local"],
                errors="coerce",
            )
            previous_observation = observation_time.groupby(
                airport_key,
                observed=True,
            ).shift(1)
            repeated_observation = (
                observation_time.notna()
                & observation_time.eq(previous_observation)
            )
            precip_source = precip_source.mask(repeated_observation)

        if not include_current:
            precip_source = precip_source.groupby(
                airport_key,
                observed=True,
            ).shift(1)

        out["precip_3h"] = (
            precip_source.groupby(airport_key, observed=True)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    if "wind_gust_kt" in out.columns:
        gust_source = pd.to_numeric(out["wind_gust_kt"], errors="coerce")
        if not include_current:
            gust_source = gust_source.groupby(
                airport_key,
                observed=True,
            ).shift(1)

        out["max_gust_3h"] = (
            gust_source.groupby(airport_key, observed=True)
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )

    return out


def scheduled_departure_timestamp(
    df: pd.DataFrame,
    date_col: str = "FlightDate",
    time_col: str = "CRSDepTime",
) -> pd.Series:
    """Return the exact naive local scheduled-departure timestamp.

    BTS stores scheduled times as local ``hhmm`` integers. A defensive value of
    2400 is interpreted as midnight at the start of the next calendar day.
    """
    for column in (date_col, time_col):
        if column not in df.columns:
            raise KeyError(f"{column!r} not in dataframe")

    t = pd.to_numeric(df[time_col], errors="coerce")
    if t.isna().any():
        raise ValueError(f"{t.isna().sum():,} rows have a non-numeric {time_col}")
    if ((t < 0) | (t > 2400)).any():
        raise ValueError(f"{time_col} outside 0-2400; check the source data")
    if (t % 100 >= 60).any():
        raise ValueError(f"{time_col} has minute values >= 60; not hhmm as assumed")

    date = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if date.isna().any():
        raise ValueError(f"{date.isna().sum():,} rows have an invalid {date_col}")

    is_2400 = t.eq(2400)
    hour = (t // 100).where(~is_2400, 0).astype("int16")
    minute = (t % 100).where(~is_2400, 0).astype("int16")

    return (
        date
        + pd.to_timedelta(is_2400.astype("int8"), unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )


def scheduled_departure_hour(
    df: pd.DataFrame,
    date_col: str = "FlightDate",
    time_col: str = "CRSDepTime",
) -> pd.Series:
    """Return the origin-airport local scheduled departure hour.

    This key is safe for origin weather stored in origin local time. It must not
    also be used for destination weather on flights crossing time zones.
    """
    return scheduled_departure_timestamp(
        df,
        date_col=date_col,
        time_col=time_col,
    ).dt.floor("h")


# --------------------------------------------------------------------------- #
# Airport timezones -- required for destination weather
# --------------------------------------------------------------------------- #
#
# Weather hours are AIRPORT-LOCAL. A flight leaving LAX at 08:00 Pacific meets
# JFK's clock at 11:00 Eastern, so joining JFK weather at "08:00" returns
# conditions three hours before the flight departed. Roughly 50% of US domestic
# flights cross a time zone, making this systematically wrong rather than
# occasionally wrong.


def load_airport_coordinates(
    path: str | Path = "data/lookups/airport_coords.csv",
) -> pd.DataFrame:
    """Load OurAirports coordinates under a stable Airport/Latitude/Longitude schema."""
    raw = pd.read_csv(path)

    candidates = {
        "Airport": ["Airport", "airport", "iata_code", "IATA", "iata"],
        "Latitude": ["Latitude", "latitude", "latitude_deg", "lat"],
        "Longitude": ["Longitude", "longitude", "longitude_deg", "lon", "lng"],
    }

    rename = {}
    for standard, options in candidates.items():
        found = next((c for c in options if c in raw.columns), None)
        if found is None:
            raise KeyError(
                f"{path} has no column for {standard}. Tried {options}; "
                f"available columns: {list(raw.columns)}"
            )
        rename[found] = standard

    coords = raw.rename(columns=rename)[["Airport", "Latitude", "Longitude"]].copy()
    coords["Airport"] = coords["Airport"].astype("string").str.strip().str.upper()
    coords["Latitude"] = pd.to_numeric(coords["Latitude"], errors="coerce")
    coords["Longitude"] = pd.to_numeric(coords["Longitude"], errors="coerce")
    coords = coords.dropna().drop_duplicates("Airport", keep="last")

    invalid = ~coords["Latitude"].between(-90, 90) | ~coords["Longitude"].between(-180, 180)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum()):,} airport coordinates are outside valid bounds")
    return coords


def build_airport_timezone_lookup(
    coords: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve airport coordinates to IANA timezone names using timezonefinder."""
    try:
        from timezonefinder import TimezoneFinder
    except ImportError as exc:
        raise ImportError(
            "E34 requires timezonefinder. Install it with `%pip install timezonefinder`."
        ) from exc

    try:
        finder = TimezoneFinder(in_memory=True)
    except TypeError:
        finder = TimezoneFinder()

    lookup = coords[["Airport", "Latitude", "Longitude"]].copy()
    lookup["timezone"] = [
        finder.timezone_at(lng=float(lon), lat=float(lat))
        for lat, lon in zip(lookup["Latitude"], lookup["Longitude"])
    ]

    unresolved = lookup["timezone"].isna()
    if unresolved.any():
        warnings.warn(
            f"{int(unresolved.sum()):,} airport timezones could not be resolved; "
            "destination weather will be missing for those airports."
        )
    return lookup


def add_destination_departure_keys(
    flights: pd.DataFrame,
    timezone_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Convert each scheduled departure instant into destination-local time."""
    required = {"Origin", "Dest", "FlightDate", "CRSDepTime"}
    missing = required - set(flights.columns)
    if missing:
        raise KeyError(f"Missing columns required for timezone conversion: {sorted(missing)}")

    timezone_map = timezone_lookup.set_index("Airport")["timezone"]
    result = flights.copy()
    departure_origin_local = scheduled_departure_timestamp(result)

    origin_tz = result["Origin"].astype("string").map(timezone_map)
    dest_tz = result["Dest"].astype("string").map(timezone_map)

    destination_local = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
    valid = origin_tz.notna() & dest_tz.notna() & departure_origin_local.notna()

    pairs = pd.DataFrame(
        {"origin_tz": origin_tz[valid], "dest_tz": dest_tz[valid]},
        index=result.index[valid],
    )

    for (origin_zone, dest_zone), idx in pairs.groupby(
        ["origin_tz", "dest_tz"], observed=True
    ).groups.items():
        localised = (
            pd.DatetimeIndex(departure_origin_local.loc[idx])
            .tz_localize(
                origin_zone,
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            .tz_convert(dest_zone)
            .tz_localize(None)
        )
        destination_local.loc[idx] = localised

    result["dest_departure_time_local"] = destination_local
    result["dest_departure_hour_local"] = destination_local.dt.floor("h")
    return result


# --------------------------------------------------------------------------- #
# Scheduled demand
# --------------------------------------------------------------------------- #


def add_schedule_demand(
    all_flights: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe scheduled airport-hour demand features.

    ``all_flights`` must include the full published schedule for the period,
    including flights later cancelled or disrupted. ``model_rows`` is the
    target-eligible subset receiving the features. Both inputs must already
    contain destination-local departure keys from
    :func:`add_destination_departure_keys`.
    """
    required_all = {
        "Origin",
        "Dest",
        "FlightDate",
        "CRSDepTime",
        "CRSElapsedTime",
        "dest_departure_time_local",
    }
    required_model = {
        "Origin",
        "Dest",
        "FlightDate",
        "CRSDepTime",
        "dest_departure_hour_local",
    }

    missing_all = sorted(required_all - set(all_flights.columns))
    missing_model = sorted(required_model - set(model_rows.columns))
    if missing_all:
        raise KeyError(f"all_flights is missing required columns: {missing_all}")
    if missing_model:
        raise KeyError(f"model_rows is missing required columns: {missing_model}")

    output_columns = {
        "origin_sched_deps_this_hour",
        "origin_sched_arrivals_this_hour",
        "origin_sched_ops_this_hour",
        "dest_sched_arrivals_this_hour",
    }
    collisions = sorted(output_columns.intersection(model_rows.columns))
    if collisions:
        raise ValueError(
            "model_rows already contains scheduled-demand columns: "
            f"{collisions}"
        )

    schedule = all_flights.copy()
    schedule["scheduled_dep_hour"] = scheduled_departure_hour(schedule)
    schedule["scheduled_arrival_hour"] = (
        pd.to_datetime(
            schedule["dest_departure_time_local"],
            errors="coerce",
        )
        + pd.to_timedelta(
            pd.to_numeric(
                schedule["CRSElapsedTime"],
                errors="coerce",
            ),
            unit="m",
        )
    ).dt.floor("h")

    departures = (
        schedule.groupby(
            ["Origin", "scheduled_dep_hour"],
            observed=True,
        )
        .size()
        .rename("origin_sched_deps_this_hour")
        .reset_index()
    )
    arrivals = (
        schedule.dropna(subset=["scheduled_arrival_hour"])
        .groupby(
            ["Dest", "scheduled_arrival_hour"],
            observed=True,
        )
        .size()
        .rename("scheduled_arrivals")
        .reset_index()
    )

    out = model_rows.copy()
    if "scheduled_dep_hour" not in out.columns:
        out["scheduled_dep_hour"] = scheduled_departure_hour(out)

    row_key = "__schedule_demand_row_id__"
    if row_key in out.columns:
        raise ValueError(f"{row_key!r} is reserved for the schedule-demand join")
    out[row_key] = np.arange(len(out), dtype=np.int64)

    out = out.merge(
        departures,
        on=["Origin", "scheduled_dep_hour"],
        how="left",
        validate="many_to_one",
    )
    if out["origin_sched_deps_this_hour"].isna().any():
        raise AssertionError(
            "origin departure-demand join left unmatched model rows"
        )

    origin_arrivals = arrivals.rename(
        columns={
            "Dest": "Origin",
            "scheduled_arrival_hour": "scheduled_dep_hour",
            "scheduled_arrivals": "origin_sched_arrivals_this_hour",
        }
    )
    out = out.merge(
        origin_arrivals,
        on=["Origin", "scheduled_dep_hour"],
        how="left",
        validate="many_to_one",
    )

    destination_arrivals = arrivals.rename(
        columns={
            "scheduled_arrival_hour": "dest_departure_hour_local",
            "scheduled_arrivals": "dest_sched_arrivals_this_hour",
        }
    )
    out = out.merge(
        destination_arrivals,
        on=["Dest", "dest_departure_hour_local"],
        how="left",
        validate="many_to_one",
    )

    count_columns = [
        "origin_sched_deps_this_hour",
        "origin_sched_arrivals_this_hour",
        "dest_sched_arrivals_this_hour",
    ]
    out[count_columns] = (
        out[count_columns]
        .fillna(0)
        .astype("int32")
    )
    out["origin_sched_ops_this_hour"] = (
        out["origin_sched_deps_this_hour"]
        + out["origin_sched_arrivals_this_hour"]
    ).astype("int32")

    out = (
        out.sort_values(row_key)
        .drop(columns=row_key)
        .reset_index(drop=True)
    )
    if len(out) != len(model_rows):
        raise AssertionError(
            "schedule-demand joins changed the row count: "
            f"{len(model_rows):,} -> {len(out):,}"
        )

    return out


def add_weather_features(
    df: pd.DataFrame,
    wx: pd.DataFrame,
    date_col: str = "FlightDate",
    time_col: str = "CRSDepTime",
    origin_col: str = "Origin",
    dest_col: str = "Dest",
    airport_col: str = "airport",
    hour_col: str = "hour",
    *,
    join_destination: bool = True,
    dest_departure_hour_col: str | None = None,
    dest_departure_time_col: str = "dest_departure_time_local",
) -> pd.DataFrame:
    """Join safe hourly weather snapshots onto flights.

    Origin weather joins on the origin-local scheduled departure hour.

    Destination weather requires a key representing that same physical instant
    in the destination airport's local timezone. Reusing the origin clock hour
    is wrong across time zones: 08:00 in New York is 05:00 in Los Angeles.

    Pass ``dest_departure_hour_col`` when destination weather is requested, or
    set ``join_destination=False`` until that timezone-aware key is available.
    """
    required_flight = {date_col, time_col, origin_col}
    if join_destination:
        required_flight.add(dest_col)
    missing_flight = sorted(required_flight - set(df.columns))
    if missing_flight:
        raise KeyError(f"flight table is missing required columns: {missing_flight}")

    missing_weather = sorted({airport_col, hour_col} - set(wx.columns))
    if missing_weather:
        raise KeyError(f"weather table is missing required columns: {missing_weather}")

    weather = wx.copy()
    weather[hour_col] = pd.to_datetime(weather[hour_col], errors="coerce")
    if weather[hour_col].isna().any():
        raise ValueError(
            f"{weather[hour_col].isna().sum():,} weather rows have an invalid "
            f"{hour_col!r}"
        )

    duplicated = weather.duplicated([airport_col, hour_col])
    if duplicated.any():
        raise ValueError(
            f"weather table has {int(duplicated.sum()):,} duplicate "
            f"({airport_col}, {hour_col}) rows"
        )

    if join_destination and dest_departure_hour_col is None:
        raise ValueError(
            "Destination weather cannot safely use the origin-local departure "
            "hour. Pass dest_departure_hour_col containing the departure instant "
            "converted to destination local time, or call with "
            "join_destination=False."
        )
    if join_destination and dest_departure_hour_col not in df.columns:
        raise KeyError(f"{dest_departure_hour_col!r} not in flight table")

    origin_cols = [c for c in ORIGIN_WEATHER_COLS if c in weather.columns]
    dest_cols = [c for c in DEST_WEATHER_COLS if c in weather.columns]
    if not origin_cols:
        raise ValueError("none of ORIGIN_WEATHER_COLS exist in the weather table")
    if join_destination and not dest_cols:
        raise ValueError("none of DEST_WEATHER_COLS exist in the weather table")

    metadata = [
        c for c in ("observation_time_local", "weather_observation_missing")
        if c in weather.columns
    ]

    # A second weather join onto a frame that already carries weather would make
    # pandas silently create ``_x``/``_y`` columns. Fail explicitly instead.
    generated_columns = {
        *(f"origin_{c}" for c in origin_cols + metadata),
        "origin_weather_age_minutes",
    }
    if join_destination:
        generated_columns.update(
            {
                *(f"dest_{c}" for c in dest_cols + metadata),
                "dest_weather_age_minutes",
            }
        )
    collisions = sorted(generated_columns.intersection(df.columns))
    if collisions:
        raise ValueError(
            "flight table already contains weather output columns; join weather "
            f"only once or drop these columns first: {collisions}"
        )

    out = df.copy()
    n_before = len(out)

    # Carry the exact timestamp through both merges. ``DataFrame.merge`` resets
    # the index, so keeping this as a column avoids index-alignment bugs when the
    # input frame has a non-default index.
    out["__scheduled_departure_local__"] = scheduled_departure_timestamp(
        out,
        date_col=date_col,
        time_col=time_col,
    )
    out["__origin_weather_hour__"] = (
        out["__scheduled_departure_local__"].dt.floor("h")
    )

    origin_weather = weather[
        [airport_col, hour_col] + origin_cols + metadata
    ].rename(
        columns={
            airport_col: origin_col,
            hour_col: "__origin_weather_hour__",
            **{c: f"origin_{c}" for c in origin_cols + metadata},
        }
    )

    out = out.merge(
        origin_weather,
        on=[origin_col, "__origin_weather_hour__"],
        how="left",
        validate="many_to_one",
    )

    observation_col = "origin_observation_time_local"
    if observation_col in out.columns:
        origin_age = (
            out["__scheduled_departure_local__"]
            - pd.to_datetime(out[observation_col], errors="coerce")
        ).dt.total_seconds().div(60)

        future = origin_age.dropna().lt(0)
        if future.any():
            raise AssertionError(
                f"{int(future.sum()):,} flights received future origin weather"
            )
        out["origin_weather_age_minutes"] = origin_age.astype("float32")

    if join_destination:
        raw_destination_hour = out[dest_departure_hour_col]
        destination_hour = pd.to_datetime(
            raw_destination_hour,
            errors="coerce",
        ).dt.floor("h")

        # ``NaT`` is a legitimate result for an unresolved airport timezone or
        # an ambiguous fall-DST wall-clock time. Preserve those rows as missing
        # destination weather, but reject non-null values that failed parsing.
        invalid_destination_hour = (
            raw_destination_hour.notna()
            & destination_hour.isna()
        )
        if invalid_destination_hour.any():
            raise ValueError(
                f"{int(invalid_destination_hour.sum()):,} non-null values in "
                f"{dest_departure_hour_col!r} could not be parsed"
            )

        out["__dest_weather_hour__"] = destination_hour

        destination_weather = weather[
            [airport_col, hour_col] + dest_cols + metadata
        ].rename(
            columns={
                airport_col: dest_col,
                hour_col: "__dest_weather_hour__",
                **{c: f"dest_{c}" for c in dest_cols + metadata},
            }
        )

        out = out.merge(
            destination_weather,
            on=[dest_col, "__dest_weather_hour__"],
            how="left",
            validate="many_to_one",
        )

    destination_observation_col = "dest_observation_time_local"
    if join_destination and destination_observation_col in out.columns:
        # Symmetric to the origin check. Compare against the EXACT destination-local
        # departure instant, never against dest_departure_hour_col -- the join key
        # cannot validate itself, and a timezone error moves key and observation
        # together, so a self-referential check would always pass.
        if dest_departure_time_col in out.columns:
            destination_reference = pd.to_datetime(
                out[dest_departure_time_col], errors="coerce"
            )
        else:
            # Fall back to the end of the keyed hour: the latest instant any flight
            # in that hour could depart. Weaker, but still catches gross errors.
            destination_reference = (
                pd.to_datetime(out[dest_departure_hour_col], errors="coerce")
                + pd.Timedelta("59min")
            )
            warnings.warn(
                f"{dest_departure_time_col!r} not found; destination leak check "
                "falls back to the end of the keyed hour. Pass the exact "
                "destination-local departure time for a strict check."
            )

        destination_age = (
            destination_reference
            - pd.to_datetime(out[destination_observation_col], errors="coerce")
        ).dt.total_seconds().div(60)

        future_destination = destination_age.dropna().lt(0)
        if future_destination.any():
            raise AssertionError(
                f"{int(future_destination.sum()):,} flights received future "
                "destination weather; check the timezone conversion"
            )
        out["dest_weather_age_minutes"] = destination_age.astype("float32")

    drop_cols = [
        "__scheduled_departure_local__",
        "__origin_weather_hour__",
    ]
    if "__dest_weather_hour__" in out.columns:
        drop_cols.append("__dest_weather_hour__")
    out = out.drop(columns=drop_cols)

    if len(out) != n_before:
        raise AssertionError(
            f"weather join changed row count: {n_before:,} -> {len(out):,}"
        )

    if "origin_weather_observation_missing" in out.columns:
        missing_rate = (
            out["origin_weather_observation_missing"]
            .fillna(1)
            .astype("float32")
            .mean()
        )
    else:
        origin_feature_cols = [f"origin_{c}" for c in origin_cols]
        missing_rate = out[origin_feature_cols].isna().all(axis=1).mean()

    print(
        f"  weather joined: {len(out):,} rows | "
        f"origin weather missing {missing_rate:.1%}"
    )
    return out
