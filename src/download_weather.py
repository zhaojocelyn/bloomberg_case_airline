"""
Download IEM ASOS hourly weather observations for all 362 origin airports,
January 2024 – May 2026.

One CSV per airport is written to data/raw/weather/{airport}.csv. Files already
on disk are skipped — re-run freely to resume after an interruption.

Timestamps are in AIRPORT LOCAL TIME (tz= parameter passed to IEM). This is
critical: BTS departure times are local, so the join in src/features.py is
local-hour to local-hour with no conversion. Requesting UTC and joining against
local time would silently shift weather by 4–8 hours. See CLAUDE.md for the
full rationale and per-airport timezone overrides.

Usage
-----
    python src/download_weather.py               # all airports
    python src/download_weather.py --force       # re-download everything
    python src/download_weather.py ORD ATL       # specific airports only
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Fields requested from IEM. Stored as-is; parsing happens in src/weather.py.
#   tmpf   — temperature (°F); converted to °C downstream
#   vsby   — visibility (statute miles)
#   sknt   — wind speed (knots)
#   gust   — wind gust (knots); "M" when calm, needs coercion
#   skyc1/skyl1 … skyc3/skyl3 — sky layer coverage + height (ft); ceiling derived
#   wxcodes — METAR weather codes; parsed for TS (thunderstorm), SN/FZRA/PL/IC
#   p01i   — precipitation last hour (inches); "T" for trace, needs coercion
IEM_FIELDS = [
    "tmpf", "vsby", "sknt", "gust",
    "skyc1", "skyl1", "skyc2", "skyl2", "skyc3", "skyl3",
    "wxcodes", "p01i",
]

# Full window matches the BTS download
DATE_START = dict(year1=2024, month1=1, day1=1)
DATE_END   = dict(year2=2026, month2=5, day2=31)

RAW_DIR    = Path("data/raw/weather")
INTERIM_DIR = Path("data/interim")

MAX_RETRIES   = 3
RETRY_BACKOFF = 10   # seconds; doubled each retry
REQUEST_PAUSE = 0.5  # seconds between airports (IEM is a public service)

# ---------------------------------------------------------------------------
# Timezone mapping
# ---------------------------------------------------------------------------

# Default IANA timezone by US state/territory abbreviation.
# Edge cases: some states span two zones — the default here is the majority
# zone, and per-airport overrides below handle the exceptions. All overrides
# are documented in CLAUDE.md.
_STATE_TZ: dict[str, str] = {
    # Eastern
    "CT": "America/New_York", "DC": "America/New_York", "DE": "America/New_York",
    "FL": "America/New_York", "GA": "America/New_York", "IN": "America/New_York",
    "KY": "America/New_York", "MA": "America/New_York", "MD": "America/New_York",
    "ME": "America/New_York", "MI": "America/New_York", "NC": "America/New_York",
    "NH": "America/New_York", "NJ": "America/New_York", "NY": "America/New_York",
    "OH": "America/New_York", "PA": "America/New_York", "RI": "America/New_York",
    "SC": "America/New_York", "VA": "America/New_York", "VT": "America/New_York",
    "WV": "America/New_York",
    # Central
    "AL": "America/Chicago", "AR": "America/Chicago", "IA": "America/Chicago",
    "IL": "America/Chicago", "KS": "America/Chicago", "LA": "America/Chicago",
    "MN": "America/Chicago", "MO": "America/Chicago", "MS": "America/Chicago",
    "ND": "America/Chicago", "NE": "America/Chicago", "OK": "America/Chicago",
    "SD": "America/Chicago", "TN": "America/Chicago", "TX": "America/Chicago",
    "WI": "America/Chicago",
    # Mountain
    "AZ": "America/Phoenix",  # no DST
    "CO": "America/Denver", "ID": "America/Denver", "MT": "America/Denver",
    "NM": "America/Denver", "UT": "America/Denver", "WY": "America/Denver",
    # Pacific
    "CA": "America/Los_Angeles", "NV": "America/Los_Angeles",
    "OR": "America/Los_Angeles", "WA": "America/Los_Angeles",
    # Alaska / Hawaii / territories
    "AK": "America/Anchorage",
    "HI": "Pacific/Honolulu",
    "PR": "America/Puerto_Rico",
    "VI": "America/St_Thomas",
    "GU": "Pacific/Guam",
    "AS": "Pacific/Pago_Pago",
    "MP": "Pacific/Saipan",
    # BTS uses "TT" for some territory flights
    "TT": "America/New_York",
}

# Per-airport overrides for airports in the non-majority timezone of their state.
# See CLAUDE.md for the full table.
_AIRPORT_TZ_OVERRIDE: dict[str, str] = {
    "PAH": "America/Chicago",      # Paducah KY — western KY is Central
    "BFF": "America/Denver",        # Scottsbluff NE — Mountain
    "DIK": "America/Denver",        # Dickinson ND — Mountain
    "XWA": "America/Denver",        # Williston ND — Mountain
    "RAP": "America/Denver",        # Rapid City SD — Mountain
    "CHA": "America/New_York",      # Chattanooga TN — Eastern
    "TRI": "America/New_York",      # Tri-Cities TN — Eastern
    "TYS": "America/New_York",      # Knoxville TN — Eastern
    "ELP": "America/Denver",        # El Paso TX — Mountain
    "LWS": "America/Los_Angeles",   # Lewiston ID — Pacific
}


def airport_timezone(airport: str, state: str) -> str:
    if airport in _AIRPORT_TZ_OVERRIDE:
        return _AIRPORT_TZ_OVERRIDE[airport]
    if state not in _STATE_TZ:
        raise ValueError(f"No timezone mapping for state {state!r} (airport {airport})")
    return _STATE_TZ[state]


# ---------------------------------------------------------------------------
# Airport list
# ---------------------------------------------------------------------------

def get_airports() -> pd.DataFrame:
    """Return DataFrame with columns [Origin, OriginState, tz] from interim parquet."""
    frames = []
    for p in sorted(INTERIM_DIR.glob("bts_*.parquet")):
        frames.append(pd.read_parquet(p, columns=["Origin", "OriginState"]))
    airports = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Origin"])
        .sort_values("Origin")
        .reset_index(drop=True)
    )
    airports["tz"] = airports.apply(
        lambda r: airport_timezone(r["Origin"], r["OriginState"]), axis=1
    )
    return airports


# ---------------------------------------------------------------------------
# Download one airport
# ---------------------------------------------------------------------------

def _raw_path(airport: str) -> Path:
    return RAW_DIR / f"{airport}.csv"


def download_airport(airport: str, tz: str, force: bool = False) -> Path:
    """Fetch IEM ASOS data for one airport and write to data/raw/weather/.

    Returns the path. Raises RuntimeError after MAX_RETRIES failures.
    Empty responses (station not in ASOS network) are written as header-only
    files so the airport is not re-requested on the next run.
    """
    dest = _raw_path(airport)
    if dest.exists() and not force:
        return dest

    params = {
        "station": airport,
        "data": IEM_FIELDS,
        "tz": tz,
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": 3,   # routine hourly METARs only; specials excluded
        **DATE_START,
        **DATE_END,
    }

    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(IEM_URL, params=params, timeout=120)
            resp.raise_for_status()

            text = resp.text
            # IEM returns an error page (HTML) rather than a 4xx when a station
            # is unknown — detect by checking whether the first non-comment line
            # looks like a CSV header.
            lines = [l for l in text.splitlines() if not l.startswith("#")]
            if lines and not lines[0].startswith("station"):
                raise RuntimeError(
                    f"{airport}: unexpected response (not CSV): {lines[0][:120]}"
                )

            dest.write_text(text, encoding="utf-8")
            return dest

        except Exception as exc:
            if attempt == MAX_RETRIES:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{airport}: failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            tqdm.write(f"    {airport} attempt {attempt} failed ({exc}), retry in {backoff}s")
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "airports", nargs="*",
        help="Airport codes to download (default: all 362 in the BTS data)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the file already exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_airports = get_airports()

    if args.airports:
        requested = set(a.upper() for a in args.airports)
        unknown = requested - set(all_airports["Origin"])
        if unknown:
            print(f"Warning: {sorted(unknown)} not found in BTS data — skipping")
        subset = all_airports[all_airports["Origin"].isin(requested)]
    else:
        subset = all_airports

    already = sum(1 for _, r in subset.iterrows()
                  if _raw_path(r["Origin"]).exists() and not args.force)
    to_fetch = len(subset) - already
    print(f"{len(subset)} airports  |  {already} already on disk  |  {to_fetch} to download")

    errors = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Airports", unit="ap"):
        airport, tz = row["Origin"], row["tz"]
        if _raw_path(airport).exists() and not args.force:
            continue
        try:
            download_airport(airport, tz, force=args.force)
        except RuntimeError as exc:
            tqdm.write(f"  ERROR: {exc}")
            errors.append(airport)
        time.sleep(REQUEST_PAUSE)

    print(f"\nDone. {to_fetch - len(errors)} downloaded, {len(errors)} failed.")
    if errors:
        print(f"Failed airports: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
