from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def _split_tickers(value: object) -> list[str]:
    if pd.isna(value):
        return []

    return [ticker.strip() for ticker in str(value).split(",") if ticker.strip()]


def load_historical_constituents(source: str | Path) -> pd.DataFrame:
    """
    Load historical S&P 500 constituents into long date/ticker format.

    Preferred input format is one row per constituent membership:

    date,ticker
    2024-01-01,A
    2024-01-01,B

    The loader also accepts snapshot-style CSVs where each date row contains
    a comma-separated list of constituents, and normalizes them to the same
    long format.
    """
    raw = pd.read_csv(source)

    if raw.empty:
        raise ValueError("historical constituents must not be empty")

    column_lookup = {column.lower().strip(): column for column in raw.columns}

    if "date" not in column_lookup:
        raise ValueError("historical constituents must contain a date column")

    date_column = column_lookup["date"]
    ticker_column = column_lookup.get("ticker")

    if ticker_column is None:
        candidates = [
            column_lookup[name]
            for name in ("tickers", "constituents", "components", "symbols")
            if name in column_lookup
        ]
        if candidates:
            ticker_column = candidates[0]
        elif len(raw.columns) >= 2:
            ticker_column = next(column for column in raw.columns if column != date_column)
        else:
            raise ValueError("historical constituents must contain a ticker column")

    constituents = raw[[date_column, ticker_column]].rename(
        columns={date_column: "date", ticker_column: "ticker"}
    )
    constituents["date"] = pd.to_datetime(constituents["date"])
    constituents["ticker"] = constituents["ticker"].map(_split_tickers)
    constituents = constituents.explode("ticker")
    constituents["ticker"] = constituents["ticker"].astype(str).str.strip()
    constituents = constituents[constituents["ticker"] != ""]
    constituents = constituents.drop_duplicates().sort_values(["date", "ticker"])

    if constituents.empty:
        raise ValueError("historical constituents contain no usable tickers")

    return constituents.reset_index(drop=True)



def load_ticker_mappings(source: str | Path | None) -> pd.DataFrame:
    """Load date-ranged ticker aliases.

    Expected columns are source_ticker,target_ticker,start,end. The end
    column is optional and inclusive when present. Missing start/end values
    make the mapping open-ended on that side.
    """
    columns = ["source_ticker", "target_ticker", "start", "end"]
    if source is None:
        return pd.DataFrame(columns=columns)

    path = Path(source)
    if not path.exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=columns)

    column_lookup = {column.lower().strip(): column for column in raw.columns}
    source_column = (
        column_lookup.get("source_ticker")
        or column_lookup.get("source")
        or column_lookup.get("old_ticker")
        or column_lookup.get("old")
    )
    target_column = (
        column_lookup.get("target_ticker")
        or column_lookup.get("target")
        or column_lookup.get("new_ticker")
        or column_lookup.get("new")
    )
    start_column = (
        column_lookup.get("start")
        or column_lookup.get("start_date")
        or column_lookup.get("from")
        or column_lookup.get("effective_date")
    )
    end_column = column_lookup.get("end") or column_lookup.get("end_date") or column_lookup.get("to")
    if source_column is None or target_column is None:
        raise ValueError(
            "ticker mapping file must contain source_ticker and target_ticker columns"
        )

    mappings = raw[[source_column, target_column]].rename(
        columns={source_column: "source_ticker", target_column: "target_ticker"}
    )
    mappings["source_ticker"] = mappings["source_ticker"].astype(str).str.strip()
    mappings["target_ticker"] = mappings["target_ticker"].astype(str).str.strip()
    mappings = mappings[
        (mappings["source_ticker"] != "")
        & (mappings["target_ticker"] != "")
        & (mappings["source_ticker"].str.lower() != "nan")
        & (mappings["target_ticker"].str.lower() != "nan")
    ]

    mappings["start"] = (
        pd.to_datetime(raw[start_column], errors="coerce")
        if start_column is not None
        else pd.NaT
    )
    mappings["end"] = (
        pd.to_datetime(raw[end_column], errors="coerce")
        if end_column is not None
        else pd.NaT
    )

    return mappings[columns].drop_duplicates().reset_index(drop=True)


def _mapping_date_mask(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    grace_days: int = 0,
) -> pd.Series:
    mask = pd.Series(True, index=index)
    effective_start = start
    if grace_days > 0 and not pd.isna(start) and pd.isna(end):
        effective_start = start - pd.Timedelta(days=grace_days)
    if not pd.isna(effective_start):
        mask &= index >= effective_start
    if not pd.isna(end):
        mask &= index <= end
    return mask


def apply_ticker_mappings(
    constituents: pd.DataFrame,
    mappings: pd.DataFrame,
) -> pd.DataFrame:
    """Apply date-ranged ticker aliases to constituent rows by row date."""
    if mappings.empty:
        return constituents

    mapped = constituents.copy()
    for row in mappings.itertuples(index=False):
        mask = mapped["ticker"].eq(row.source_ticker)
        if not pd.isna(row.start):
            mask &= mapped["date"] >= row.start
        if not pd.isna(row.end):
            mask &= mapped["date"] <= row.end
        mapped.loc[mask, "ticker"] = row.target_ticker

    mapped = mapped.drop_duplicates().sort_values(["date", "ticker"])
    return mapped.reset_index(drop=True)


def apply_ticker_mappings_to_membership(
    membership: pd.DataFrame,
    mappings: pd.DataFrame,
    grace_days: int = 0,
) -> pd.DataFrame:
    """Apply date-ranged ticker aliases to a daily membership matrix.

    grace_days stretches open-ended replacement mappings backward. Closed
    mappings keep their exact dates.
    """
    if mappings.empty:
        return membership

    mapped = membership.copy()
    mapped.index = pd.DatetimeIndex(mapped.index)

    for row in mappings.itertuples(index=False):
        source = row.source_ticker
        target = row.target_ticker
        if source not in mapped.columns:
            continue
        if target not in mapped.columns:
            mapped[target] = False

        date_mask = _mapping_date_mask(mapped.index, row.start, row.end, grace_days)
        active = mapped[source].fillna(False) & date_mask
        if active.any():
            mapped.loc[active, target] = True
            mapped.loc[active, source] = False

    mapped = mapped.loc[:, mapped.any(axis=0)]
    return mapped.reindex(sorted(mapped.columns), axis=1)

def tickers_for_period(
    constituents: pd.DataFrame,
    start: str,
    end: str | None = None,
) -> list[str]:
    """Return every ticker that appears in a constituent snapshot for a period."""
    if constituents.empty:
        raise ValueError("historical constituents must not be empty")

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end is not None else constituents["date"].max()

    snapshot_dates = pd.Index(constituents["date"].drop_duplicates().sort_values())
    start_position = snapshot_dates.searchsorted(start_date, side="right") - 1

    included_dates = snapshot_dates[(snapshot_dates >= start_date) & (snapshot_dates <= end_date)]
    if start_position >= 0:
        included_dates = included_dates.union(pd.Index([snapshot_dates[start_position]]))

    if included_dates.empty:
        raise ValueError("no constituent snapshots found for requested period")

    tickers = constituents[constituents["date"].isin(included_dates)]["ticker"]
    return sorted(tickers.drop_duplicates())


def constituent_membership_matrix(
    constituents: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
) -> pd.DataFrame:
    """Align constituent snapshots to trading dates as a boolean matrix."""
    trading_dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()

    if trading_dates.empty:
        raise ValueError("dates must not be empty")

    snapshots = {
        date: sorted(group["ticker"].drop_duplicates())
        for date, group in constituents.groupby("date", sort=True)
    }
    snapshot_dates = pd.DatetimeIndex(sorted(snapshots))

    first_position = snapshot_dates.searchsorted(trading_dates[0], side="right") - 1
    if first_position < 0:
        raise ValueError("price history starts before first constituent snapshot")

    active_by_date: dict[pd.Timestamp, set[str]] = {}
    all_tickers: set[str] = set()

    for date in trading_dates:
        position = snapshot_dates.searchsorted(date, side="right") - 1
        if position < 0:
            active = set[str]()
        else:
            active = set(snapshots[snapshot_dates[position]])
        active_by_date[date] = active
        all_tickers.update(active)

    membership = pd.DataFrame(False, index=trading_dates, columns=sorted(all_tickers))

    for date, active in active_by_date.items():
        if active:
            membership.loc[date, list(active)] = True

    return membership


def membership_date_ranges(membership: pd.DataFrame) -> pd.DataFrame:
    """Return each ticker's first and last active membership date."""
    if membership.empty:
        raise ValueError("membership must not be empty")

    ranges: list[dict[str, pd.Timestamp | str]] = []

    for ticker in membership.columns:
        active_dates = membership.index[membership[ticker].fillna(False)]
        if active_dates.empty:
            continue

        ranges.append(
            {
                "ticker": ticker,
                "start": active_dates.min(),
                "end": active_dates.max(),
            }
        )

    if not ranges:
        raise ValueError("membership contains no active tickers")

    return pd.DataFrame(ranges).sort_values("ticker").reset_index(drop=True)
