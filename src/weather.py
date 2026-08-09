"""
Parse raw IEM ASOS CSVs -> data/interim/weather_hourly.parquet.

Steps
-----
1. Read per-airport CSVs from data/raw/weather/ (one per IATA code).
2. Clean: coerce "M" to NaN, "T" to 0 (trace precip), strip whitespace,
   and remove exact duplicate observations.
3. Derive ceiling_ft (lowest BKN/OVC/VV layer; 99999 if unlimited).
4. Parse wxcodes for thunderstorm and frozen_precip flags.
5. Convert units: degF to degC and precipitation inches to millimetres.
6. Build one start-of-hour snapshot per airport. Each snapshot uses only the
   latest observation available at or before that hour; observations from later
   in the hour are never moved backward. This avoids weather leakage.
7. Derive flight_category (VFR/MVFR/IFR/LIFR) via features.py.
8. Compute rolling history via features.py::add_weather_history.
9. Write parquet.

Rolling MUST run on the weather table before any join to flights. An evenly
spaced hourly series is required; flight rows are not evenly spaced, so
"previous 3 rows" is not "previous 3 hours" after the join.

Timestamp convention
--------------------
The IEM downloads used by this project are expected to contain airport-local,
naive timestamps in the ``valid`` column. The output ``hour`` is therefore also
airport-local and naive. A row at 08:00 is a snapshot containing the latest
observation at or before 08:00 (commonly the 07:51 METAR), never the 08:51
observation. Join a flight to its FLOORED local scheduled-departure hour.

For cross-time-zone work or exact observation-age calculations, retain the
``observation_time_local`` column and use an airport-specific timezone mapping
when constructing UTC timestamps downstream.

Usage
-----
    python src/weather.py
    python src/weather.py --raw-dir data/raw/weather \
        --out data/interim/weather_hourly.parquet
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# features.py lives in the same package; allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent))
from features import add_weather_history, derive_flight_category

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw/weather")
OUT_PATH = Path("data/interim/weather_hourly.parquet")

# Ceiling: sky coverage codes that define a ceiling layer.
_CEILING_CODES = {"BKN", "OVC", "VV"}

# wxcodes tokens that indicate frozen precipitation or a freezing obstruction.
_FROZEN_TOKENS = {
    "SN",
    "FZRA",
    "FZDZ",
    "PL",
    "PE",
    "GS",
    "IC",
    "FZFG",
    "BLSN",
}

# Full window -- must match the BTS download.
_GRID_START = "2024-01-01 00:00"
_GRID_END = "2026-05-31 23:00"

# A routine airport observation is commonly about 9 minutes old at the next
# top-of-hour snapshot. Allow one missed routine report, but do not carry stale
# weather forward indefinitely.
_MAX_SNAPSHOT_AGE = pd.Timedelta("90min")

_CORE_COLUMNS = {
    "valid",
    "tmpf",
    "vsby",
    "sknt",
    "gust",
    "p01i",
    "wxcodes",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _derive_ceiling(df: pd.DataFrame) -> pd.Series:
    """Return the lowest BKN/OVC/VV layer height in feet.

    Returns 99999 when sky conditions were reported but no ceiling-producing
    layer is present (an unlimited ceiling). Returns NaN only when every sky
    coverage field is missing, which indicates that the ceiling cannot be
    determined from the row.
    """
    layer_heights: list[pd.Series] = []
    sky_columns: list[pd.Series] = []

    # IEM can provide up to four cloud layers. Missing fourth-layer columns are
    # tolerated because older or narrower extracts may omit them.
    for i in range(1, 5):
        cover_col = f"skyc{i}"
        height_col = f"skyl{i}"

        if cover_col not in df.columns:
            cover = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            cover = df[cover_col].astype("string").str.strip().str.upper()

        if height_col not in df.columns:
            height = pd.Series(np.nan, index=df.index, dtype="float64")
        else:
            height = pd.to_numeric(df[height_col], errors="coerce")

        sky_columns.append(cover)
        layer_heights.append(height.where(cover.isin(_CEILING_CODES)))

    stacked_heights = pd.concat(layer_heights, axis=1)
    ceiling = stacked_heights.min(axis=1)

    stacked_sky = pd.concat(sky_columns, axis=1)
    has_sky_report = stacked_sky.notna().any(axis=1)

    # Sky was reported but there is no BKN/OVC/VV layer: unlimited ceiling.
    ceiling = ceiling.where(ceiling.notna() | ~has_sky_report, 99999.0)
    return ceiling.astype("float32")


def _parse_wxcodes(codes: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return thunderstorm and frozen-precipitation flags from METAR codes.

    The thunderstorm check handles common forms such as TSRA, -TSRA, +TSRA,
    and VCTS. A missing ``wxcodes`` value on an otherwise present observation
    means no significant weather was encoded and is therefore False. Rows
    inserted later by the hourly-grid step remain missing.
    """

    def _check(value: object) -> tuple[bool, bool]:
        if pd.isna(value):
            return False, False

        groups = str(value).strip().upper().split()
        thunderstorm = any("TS" in group for group in groups)
        frozen = any(
            token in group
            for group in groups
            for token in _FROZEN_TOKENS
        )
        return thunderstorm, frozen

    parsed = codes.map(_check)
    thunderstorm = parsed.map(lambda pair: pair[0]).astype("bool")
    frozen_precip = parsed.map(lambda pair: pair[1]).astype("bool")
    return thunderstorm, frozen_precip


def parse_one(path: Path) -> pd.DataFrame:
    """Parse one airport CSV into cleaned, exact observations.

    Returned timestamps remain airport-local and naive because that is the
    convention expected from this project's IEM downloads. Hourly snapshots are
    created later by :func:`reindex_to_grid`.
    """
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    if len(lines) < 2:
        return pd.DataFrame()

    df = pd.read_csv(
        io.StringIO("\n".join(lines)),
        dtype=str,
        na_values=["M"],
        keep_default_na=True,
    )
    if df.empty:
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    missing_core = sorted(_CORE_COLUMNS - set(df.columns))
    if missing_core:
        raise ValueError(
            f"{path.name} is missing required IEM columns: {missing_core}"
        )

    # Strip whitespace without converting missing values to literal strings.
    string_columns = df.select_dtypes(include="object").columns
    for column in string_columns:
        df[column] = df[column].str.strip()

    df["observation_time_local"] = pd.to_datetime(
        df["valid"],
        errors="coerce",
    )
    df["airport"] = path.stem.upper()
    df = df.dropna(subset=["observation_time_local"])

    # Remove exact duplicate observations caused by overlapping downloads. Do
    # not collapse all observations within an hour: special METARs may contain
    # useful changes and the snapshot step must choose only information already
    # available at each top of hour.
    df = (
        df.sort_values("observation_time_local")
        .drop_duplicates(
            subset=["airport", "observation_time_local"],
            keep="last",
        )
    )

    # Temperature: degF -> degC.
    tmpf = pd.to_numeric(df["tmpf"], errors="coerce")
    df["temp_c"] = ((tmpf - 32.0) * 5.0 / 9.0).astype("float32")

    # Visibility is already in statute miles.
    df["visibility_sm"] = pd.to_numeric(
        df["vsby"], errors="coerce"
    ).astype("float32")

    # Wind speed and gust are already in knots.
    df["wind_speed_kt"] = pd.to_numeric(
        df["sknt"], errors="coerce"
    ).astype("float32")
    df["wind_gust_kt"] = pd.to_numeric(
        df["gust"], errors="coerce"
    ).astype("float32")

    # Precipitation: trace -> 0.0 inches, then inches -> millimetres.
    precip_text = df["p01i"].astype("string").str.upper().replace("T", "0")
    precip_in = pd.to_numeric(precip_text, errors="coerce")
    df["precip_mm"] = (precip_in * 25.4).astype("float32")

    df["ceiling_ft"] = _derive_ceiling(df)
    df["ceiling_unlimited"] = (df["ceiling_ft"] == 99999).astype("int8")

    df["thunderstorm"], df["frozen_precip"] = _parse_wxcodes(
        df["wxcodes"]
    )

    keep = [
        "airport",
        "observation_time_local",
        "temp_c",
        "visibility_sm",
        "wind_speed_kt",
        "wind_gust_kt",
        "ceiling_ft",
        "ceiling_unlimited",
        "precip_mm",
        "thunderstorm",
        "frozen_precip",
    ]
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full-grid snapshots
# ---------------------------------------------------------------------------


def reindex_to_grid(
    df: pd.DataFrame,
    max_snapshot_age: pd.Timedelta = _MAX_SNAPSHOT_AGE,
) -> pd.DataFrame:
    """Create a contiguous start-of-hour weather snapshot per airport.

    Each row at time ``hour`` receives the latest observation with
    ``observation_time_local <= hour``. A later observation from within that
    hour is never moved backward, eliminating the leakage caused by flooring an
    08:51 observation to 08:00.

    Observations older than ``max_snapshot_age`` are not carried forward. Such
    grid rows retain NaN weather values so XGBoost can handle the missingness.
    """
    full_hours = pd.date_range(_GRID_START, _GRID_END, freq="h")
    airport_frames: list[pd.DataFrame] = []

    grouped = df.groupby("airport", sort=True, observed=True)
    for airport, observations in tqdm(
        grouped,
        total=df["airport"].nunique(),
        unit="ap",
        desc="  grid",
    ):
        observations = observations.sort_values(
            "observation_time_local"
        ).drop(columns="airport")

        grid = pd.DataFrame({"hour": full_hours})
        snapshot = pd.merge_asof(
            grid,
            observations,
            left_on="hour",
            right_on="observation_time_local",
            direction="backward",
            tolerance=max_snapshot_age,
            allow_exact_matches=True,
        )
        snapshot.insert(0, "airport", airport)
        airport_frames.append(snapshot)

    result = pd.concat(airport_frames, ignore_index=True)

    # This is snapshot age at the top of the hour. After joining to flights,
    # calculate the exact weather age using the flight's scheduled timestamp.
    result["snapshot_age_minutes"] = (
        result["hour"] - result["observation_time_local"]
    ).dt.total_seconds().div(60).astype("float32")

    result["weather_observation_missing"] = (
        result["observation_time_local"].isna().astype("int8")
    )

    observed_ages = result["snapshot_age_minutes"].dropna()
    if not observed_ages.ge(0).all():
        raise AssertionError("A weather snapshot contains a future observation.")
    if not observed_ages.le(max_snapshot_age.total_seconds() / 60).all():
        raise AssertionError("A weather snapshot exceeds the age tolerance.")

    return result


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_weather(
    raw_dir: Path = RAW_DIR,
    out_path: Path = OUT_PATH,
) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}")

    print(f"Parsing {len(csv_files)} airport CSVs ...")
    frames: list[pd.DataFrame] = []
    unusable_airports: list[str] = []

    for path in tqdm(csv_files, unit="ap", desc="  parse"):
        parsed = parse_one(path)
        if parsed.empty:
            unusable_airports.append(path.stem.upper())
        else:
            frames.append(parsed)

    if not frames:
        raise ValueError("No usable weather observations were parsed.")

    if unusable_airports:
        print(
            "  Airports with no usable observations: "
            + ", ".join(unusable_airports)
        )

    print("Concatenating exact observations ...")
    observations = pd.concat(frames, ignore_index=True)
    observations = observations.sort_values(
        ["airport", "observation_time_local"]
    ).reset_index(drop=True)
    print(f"  Exact observation rows: {len(observations):,}")

    print("Building leakage-safe hourly snapshots ...")
    df = reindex_to_grid(observations)

    hours_per_airport = len(pd.date_range(_GRID_START, _GRID_END, freq="h"))
    expected_rows = observations["airport"].nunique() * hours_per_airport
    print(f"  Grid rows: {len(df):,} (expected {expected_rows:,})")

    if len(df) != expected_rows:
        raise AssertionError(
            f"Unexpected grid size: {len(df):,} != {expected_rows:,}"
        )
    if df.duplicated(["airport", "hour"]).any():
        raise AssertionError("Duplicate airport-hour rows remain after gridding.")
    if df["hour"].isna().any():
        raise AssertionError("The hourly grid contains null timestamps.")

    print("Deriving flight_category ...")
    df["flight_category"] = derive_flight_category(
        df["ceiling_ft"],
        df["visibility_sm"],
    )

    valid_categories = {"VFR", "MVFR", "IFR", "LIFR"}
    observed_categories = set(df["flight_category"].dropna().astype(str).unique())
    unexpected_categories = observed_categories - valid_categories
    if unexpected_categories:
        raise AssertionError(
            f"Unexpected flight categories: {sorted(unexpected_categories)}"
        )

    # Rolling history must run on a sorted, evenly spaced airport-hour table.
    print("Computing rolling history ...")
    df = df.sort_values(["airport", "hour"]).reset_index(drop=True)
    df = add_weather_history(
        df,
        window=3,
        airport_col="airport",
        hour_col="hour",
    )

    # Preserve the distinction between "observed false" and "no observation".
    # Exact observations have bool values; rows with no snapshot remain <NA>.
    for column in ("thunderstorm", "frozen_precip"):
        df[column] = df[column].astype("boolean").astype("Int8")

    col_order = [
        "airport",
        "hour",
        "observation_time_local",
        "snapshot_age_minutes",
        "weather_observation_missing",
        "visibility_sm",
        "ceiling_ft",
        "ceiling_unlimited",
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
    col_order = [column for column in col_order if column in df.columns]
    df = df[col_order]

    print(f"Writing {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        out_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    size_mb = out_path.stat().st_size / 1e6
    print(
        f"\nDone. {len(df):,} rows x {len(df.columns)} cols "
        f"-> {size_mb:.0f} MB"
    )
    print("\nNull rates (columns with >0 nulls):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    print((nulls / len(df) * 100).round(1).rename("pct_null").to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    build_weather(raw_dir=args.raw_dir, out_path=args.out)


if __name__ == "__main__":
    main()
