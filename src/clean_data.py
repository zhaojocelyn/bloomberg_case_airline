"""
Revised target construction and split loading for the BTS flight-delay dataset.

Filename: ``clean_data_revised.py``. Use this module with
``features_revised.py`` and ``03_modeling_stage1_revised.py`` to avoid mixing
older drafts with the consolidated target pipeline.

Public functions
----------------
build_target(frame)
    Recover arrival-delay targets for diversions that reached their destination
    and add consistent eligibility flags.

select_arrival_target_rows(frame, require_minutes=False)
    Return flights eligible for arrival-delay modeling.

load_split(split, columns=None, model_only=False)
    Load one temporal split from ``data/interim``, apply ``build_target``, and
    optionally keep only model-eligible rows.

Split definitions
-----------------
    train : Jan-Dec 2024
    val   : Jan-Dec 2025
    test  : Jan-May 2026
    all   : Jan 2024-May 2026
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "INTERIM_DIR",
    "build_target",
    "select_arrival_target_rows",
    "load_split",
]

INTERIM_DIR = Path("data/interim")

_SPLIT_MONTHS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "train": ((2024, 1), (2024, 12)),
    "val": ((2025, 1), (2025, 12)),
    "test": ((2026, 1), (2026, 5)),
    "all": ((2024, 1), (2026, 5)),
}

_TARGET_COLS: tuple[str, ...] = (
    "Cancelled",
    "Diverted",
    "DivReachedDest",
    "DivArrDelay",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
)


# --------------------------------------------------------------------------- #
# Target construction
# --------------------------------------------------------------------------- #


def build_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover arrival-delay targets and add eligibility flags.

    Diversions are recoverable only when ``DivReachedDest == 1`` and
    ``DivArrDelay`` is present. Cancellations, unrecovered diversions, and rows
    without ``ArrDel15`` are not eligible for the arrival-delay model.
    """
    missing = sorted(set(_TARGET_COLS) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing target columns: {missing}")

    out = frame.copy()

    recovered = (
        out["Diverted"].eq(1)
        & out["DivReachedDest"].eq(1)
        & out["DivArrDelay"].notna()
    )
    out.loc[recovered, "ArrDelay"] = out.loc[recovered, "DivArrDelay"]
    out.loc[recovered, "ArrDelayMinutes"] = (
        out.loc[recovered, "DivArrDelay"].clip(lower=0)
    )
    out.loc[recovered, "ArrDel15"] = (
        out.loc[recovered, "DivArrDelay"].ge(15).astype("float32")
    )

    out["target_recovered_from_diversion"] = recovered
    out["nonarrival_disruption"] = (
        out["Cancelled"].eq(1)
        | (
            out["Diverted"].eq(1)
            & ~out["DivReachedDest"].eq(1)
        )
    )
    out["model_eligible"] = (
        ~out["nonarrival_disruption"]
        & out["ArrDel15"].notna()
    )

    return out


def select_arrival_target_rows(
    frame: pd.DataFrame,
    *,
    require_minutes: bool = False,
) -> pd.DataFrame:
    """Return rows eligible for arrival-delay modeling."""
    required = {"model_eligible", "ArrDel15"}
    if require_minutes:
        required.add("ArrDelayMinutes")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing target-selection columns: {missing}")

    mask = frame["model_eligible"].fillna(False)
    if require_minutes:
        mask &= frame["ArrDelayMinutes"].notna()

    out = frame.loc[mask].copy()
    out["ArrDel15"] = out["ArrDel15"].astype("int8")
    return out


# --------------------------------------------------------------------------- #
# Split loader
# --------------------------------------------------------------------------- #


def load_split(
    split: str,
    columns: list[str] | None = None,
    model_only: bool = False,
) -> pd.DataFrame:
    """Load a temporal split and apply the shared target definition.

    Target columns omitted from ``columns`` are loaded temporarily and removed
    after target construction. Generated audit columns remain available.
    """
    if split not in _SPLIT_MONTHS:
        raise ValueError(
            f"split must be one of {list(_SPLIT_MONTHS)}; got {split!r}"
        )

    (start_year, start_month), (end_year, end_month) = _SPLIT_MONTHS[split]
    paths: list[Path] = []

    for path in sorted(INTERIM_DIR.glob("bts_*.parquet")):
        parts = path.stem.split("_")
        if len(parts) != 3:
            continue
        year, month = int(parts[1]), int(parts[2])
        if (start_year, start_month) <= (year, month) <= (end_year, end_month):
            paths.append(path)

    if not paths:
        raise FileNotFoundError(
            f"No parquet files found for split {split!r} in {INTERIM_DIR}. "
            "Run src/download_bts.py first."
        )

    if columns is None:
        read_columns: list[str] | None = None
        temporary_columns: set[str] = set()
    else:
        requested_columns = list(dict.fromkeys(columns))
        temporary_columns = set(_TARGET_COLS) - set(requested_columns)
        read_columns = list(
            dict.fromkeys([*requested_columns, *_TARGET_COLS])
        )

    frames = [
        pd.read_parquet(path, columns=read_columns)
        for path in paths
    ]
    out = build_target(pd.concat(frames, ignore_index=True))

    if model_only:
        out = select_arrival_target_rows(out)

    if temporary_columns:
        out = out.drop(
            columns=[
                column
                for column in temporary_columns
                if column in out.columns
            ]
        )

    return out.reset_index(drop=True)
