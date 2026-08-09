"""
Download BTS Reporting Carrier On-Time Performance data (Jan 2024 – May 2026),
convert each month to Parquet with the selected column subset, and produce a
stratified 50k-row sample in data/sample/.

Usage
-----
    python src/download_bts.py               # download + convert + sample
    python src/download_bts.py --no-sample   # skip sample step
    python src/download_bts.py --sample-only # rebuild sample from existing parquet

The script is idempotent: zip files and parquet files already present are
skipped. Use --force to re-download/re-convert.
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Jan 2024 – May 2026 inclusive
MONTHS = [
    (y, m)
    for y in range(2024, 2027)
    for m in range(1, 13)
    if (y, m) <= (2026, 5)
]

# Column subset from CLAUDE.md
KEEP_COLS = [
    # Time
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    # Airline
    "Reporting_Airline", "DOT_ID_Reporting_Airline",
    "Tail_Number", "Flight_Number_Reporting_Airline",
    # Route
    "Origin", "OriginAirportSeqID", "OriginCityName", "OriginState",
    "Dest", "DestAirportSeqID", "DestCityName", "DestState",
    # Departure
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepTimeBlk", "TaxiOut", "WheelsOff",
    # Arrival
    "WheelsOn", "TaxiIn", "CRSArrTime", "ArrTime",
    "ArrDelay", "ArrDelayMinutes", "ArrDel15", "ArrTimeBlk",
    # Status
    "Cancelled", "CancellationCode", "Diverted",
    # Diverted summary (3 cols — needed because ArrDelay is NULL for diverted flights)
    "DivAirportLandings", "DivReachedDest", "DivArrDelay",
    # Summary
    "CRSElapsedTime", "ActualElapsedTime", "AirTime",
    "Flights", "Distance", "DistanceGroup",
    # Delay causes (NULL for flights delayed <15 min — missing by design, not error)
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
SAMPLE_DIR = Path("data/sample")
SAMPLE_ROWS = 50_000
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds between retries


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def zip_path(year: int, month: int) -> Path:
    return RAW_DIR / f"bts_{year}_{month}.zip"


def parquet_path(year: int, month: int) -> Path:
    return INTERIM_DIR / f"bts_{year}_{month:02d}.parquet"


def download_month(year: int, month: int, force: bool = False) -> Path:
    dest = zip_path(year, month)
    if dest.exists() and not force:
        return dest

    url = BASE_URL.format(year=year, month=month)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {year}-{month:02d}",
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
            return dest
        except Exception as exc:
            if attempt == MAX_RETRIES:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to download {year}-{month:02d} after {MAX_RETRIES} attempts"
                ) from exc
            print(f"    attempt {attempt} failed ({exc}), retrying in {RETRY_BACKOFF}s…")
            time.sleep(RETRY_BACKOFF)


# ---------------------------------------------------------------------------
# Convert to Parquet
# ---------------------------------------------------------------------------

def csv_name_in_zip(zf: zipfile.ZipFile) -> str:
    return next(n for n in zf.namelist() if n.endswith(".csv"))


def convert_month(year: int, month: int, force: bool = False) -> Path:
    src = zip_path(year, month)
    dest = parquet_path(year, month)
    if dest.exists() and not force:
        return dest
    if not src.exists():
        raise FileNotFoundError(f"Raw zip missing: {src}")

    with zipfile.ZipFile(src) as zf:
        with zf.open(csv_name_in_zip(zf)) as f:
            df = pd.read_csv(f, low_memory=False)

    # Drop the trailing empty column that comes from the BTS trailing comma
    df = df[[c for c in df.columns if c.strip()]]

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{year}-{month:02d}: expected columns not found: {missing}")

    df = df[KEEP_COLS]

    # Downcast float64 delay columns to float32 to halve memory / disk usage
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    # FlightDate as date type
    df["FlightDate"] = pd.to_datetime(df["FlightDate"]).dt.date

    df.to_parquet(dest, index=False, engine="pyarrow", compression="snappy")
    return dest


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def make_sample(force: bool = False) -> Path:
    dest = SAMPLE_DIR / "flights_sample.parquet"
    if dest.exists() and not force:
        return dest

    parquets = sorted(INTERIM_DIR.glob("bts_*.parquet"))
    if not parquets:
        raise FileNotFoundError("No parquet files found in data/interim/")

    # Proportional sample: each month contributes rows proportional to its size
    frames = []
    sizes = []
    for p in parquets:
        df = pd.read_parquet(p)
        sizes.append(len(df))
        frames.append(df)

    total = sum(sizes)
    sample_parts = []
    for df, n in zip(frames, sizes):
        frac = SAMPLE_ROWS / total  # same fraction from every month
        # groupby().sample(frac=...) preserves the groupby column (pandas 2.x)
        sample_parts.append(
            df.groupby("Reporting_Airline", group_keys=False)
              .sample(frac=frac, random_state=42)
        )

    sample = pd.concat(sample_parts, ignore_index=True)
    # Trim to exactly SAMPLE_ROWS if rounding pushed us over
    if len(sample) > SAMPLE_ROWS:
        sample = sample.sample(n=SAMPLE_ROWS, random_state=42)

    sample.to_parquet(dest, index=False, engine="pyarrow", compression="snappy")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-convert even if files exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.sample_only:
        print(f"Downloading and converting {len(MONTHS)} months "
              f"({MONTHS[0][0]}-{MONTHS[0][1]:02d} → "
              f"{MONTHS[-1][0]}-{MONTHS[-1][1]:02d})")
        for year, month in tqdm(MONTHS, desc="Months", unit="mo"):
            pq = parquet_path(year, month)
            zp = zip_path(year, month)

            if pq.exists() and not args.force:
                tqdm.write(f"  {year}-{month:02d}  parquet exists, skipping")
                continue

            if not zp.exists() or args.force:
                tqdm.write(f"  {year}-{month:02d}  downloading…")
                download_month(year, month, force=args.force)
            else:
                tqdm.write(f"  {year}-{month:02d}  zip exists, converting…")

            tqdm.write(f"  {year}-{month:02d}  converting to parquet…")
            convert_month(year, month, force=args.force)

    if not args.no_sample:
        print("\nBuilding 50k-row stratified sample…")
        path = make_sample(force=args.sample_only or args.force)
        print(f"  Sample written to {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
