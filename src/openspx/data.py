from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class InputUsageReport:
    """Collect input-file usage counts for one CLI run."""

    records: list[dict[str, object]] = field(default_factory=list)

    def add(self, stage: str, input_type: str, name: str, rows: int = 0, note: str = "") -> None:
        self.records.append({"stage": stage, "input_type": input_type, "name": name, "rows": int(rows), "note": note})

    def summary(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["stage", "input_type", "files", "rows"])
        data = pd.DataFrame(self.records)
        return (
            data.groupby(["stage", "input_type"], dropna=False)
            .agg(files=("name", "nunique"), rows=("rows", "sum"))
            .reset_index()
            .sort_values(["stage", "input_type"])
        )


def _report_series(report: InputUsageReport | None, stage: str, input_type: str, name: str, series: pd.Series | None, note: str = "") -> None:
    if report is not None and series is not None:
        report.add(stage, input_type, name, rows=int(series.dropna().shape[0]), note=note)


def _report_value(report: InputUsageReport | None, stage: str, input_type: str, name: str, note: str = "") -> None:
    if report is not None:
        report.add(stage, input_type, name, rows=1, note=note)


def format_input_usage_report(report: InputUsageReport) -> str:
    summary = report.summary()
    if summary.empty:
        return "No input-file usage was recorded."
    return summary.to_string(index=False)


def _display_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _display_range(start: object, end: object | None) -> tuple[str, str | None]:
    return (_display_date(start), None if end is None else _display_date(end))


def _range_text(date_range: tuple[str, str | None]) -> str:
    start, end = date_range
    return f"{start} to {end or 'open-ended'}"


class MissingInputDataError(ValueError):
    """Raised when required local CSV inputs are missing or unusable."""

    def __init__(
        self,
        price_tickers: list[str] | None = None,
        market_cap_tickers: list[str] | None = None,
        price_ranges: dict[str, tuple[str, str | None]] | None = None,
        market_cap_ranges: dict[str, tuple[str, str | None]] | None = None,
    ) -> None:
        self.price_tickers = sorted(set(price_tickers or []))
        self.market_cap_tickers = sorted(set(market_cap_tickers or []))
        self.price_ranges = price_ranges or {}
        self.market_cap_ranges = market_cap_ranges or {}
        parts = []
        if self.price_tickers:
            labels = []
            for ticker in self.price_tickers:
                label = ticker
                if ticker in self.price_ranges:
                    label += f" ({_range_text(self.price_ranges[ticker])})"
                labels.append(label)
            parts.append("missing price data for " f"{len(self.price_tickers)} ticker(s): " + ", ".join(labels))
        if self.market_cap_tickers:
            labels = []
            for ticker in self.market_cap_tickers:
                label = ticker
                if ticker in self.market_cap_ranges:
                    label += f" ({_range_text(self.market_cap_ranges[ticker])})"
                labels.append(label)
            parts.append("missing market-cap or shares-outstanding data for " f"{len(self.market_cap_tickers)} ticker(s): " + ", ".join(labels))
        super().__init__("; ".join(parts))


def _data_file_path(ticker: str, data_dir: str | Path, subdir: str) -> Path:
    base_dir = Path(data_dir)
    nested_path = base_dir / subdir / f"{ticker}.csv"
    if nested_path.exists():
        return nested_path
    return base_dir / f"{ticker}.csv"


def _date_column(data: pd.DataFrame) -> str:
    column_lookup = {column.lower().strip(): column for column in data.columns}
    return column_lookup.get("date", data.columns[0])


def _numeric_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    column_lookup = {column.lower().strip(): column for column in data.columns}
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]
    numeric_columns = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
    return numeric_columns[-1] if numeric_columns else None


def _series_from_file(path: str | Path, ticker: str, candidates: tuple[str, ...]) -> pd.Series | None:
    data = pd.read_csv(path)
    if data.empty:
        return None
    date_column = _date_column(data)
    value_column = _numeric_column(data, candidates)
    if value_column is None:
        return None
    series = pd.Series(pd.to_numeric(data[value_column], errors="coerce").to_numpy(), index=pd.to_datetime(data[date_column], errors="coerce"), name=ticker)
    series = series[series.index.notna()].sort_index().dropna()
    return None if series.empty else series


def _filter_and_fill_daily(series: pd.Series, start: str, end: str | None) -> pd.Series | None:
    series = series.sort_index().dropna()
    if series.empty:
        return None
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end is not None else series.index.max() + pd.Timedelta(days=1)
    requested_index = pd.date_range(start_date, end_date - pd.Timedelta(days=1), freq="D")
    if requested_index.empty:
        return None
    series = series.reindex(series.index.union(requested_index)).ffill()
    series = series.reindex(requested_index).dropna()
    return None if series.empty else series


def load_local_price_series(ticker: str, start: str, end: str | None, local_prices_dir: str | Path) -> pd.Series | None:
    path = _data_file_path(ticker, local_prices_dir, "price")
    if not path.exists():
        return None

    data = pd.read_csv(path)
    if data.empty:
        return None
    column_lookup = {column.lower().strip().replace(" ", ""): column for column in data.columns}
    has_close = "close" in column_lookup
    has_adj_close = "adjclose" in column_lookup
    if has_close and has_adj_close:
        warnings.warn(
            f"Price file {path} contains both Close and Adj Close; using Close for price-index contribution analysis.",
            RuntimeWarning,
            stacklevel=2,
        )
    if has_close:
        value_column = column_lookup["close"]
    elif "price" in column_lookup:
        value_column = column_lookup["price"]
    elif has_adj_close:
        value_column = column_lookup["adjclose"]
        warnings.warn(
            f"Price file {path} uses Adj Close. Adjusted prices may include dividend adjustments and may not align with a price-index target.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        raise ValueError(f"Price file {path} must contain Date and Close columns")

    date_column = _date_column(data)
    raw_series = pd.Series(
        pd.to_numeric(data[value_column], errors="coerce").to_numpy(),
        index=pd.to_datetime(data[date_column], errors="coerce"),
        name=ticker,
    )
    raw_series = raw_series[raw_series.index.notna()].sort_index().dropna()
    if raw_series.empty:
        return None

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end is not None else raw_series.index.max() + pd.Timedelta(days=1)
    business_days = pd.bdate_range(start_date, end_date - pd.Timedelta(days=1))
    raw_days = pd.DatetimeIndex(raw_series.index.normalize().unique())
    missing_business_days = business_days.difference(raw_days)
    if len(business_days) and len(missing_business_days):
        warnings.warn(
            f"Price file {path} is missing {len(missing_business_days)} business-day observation(s) in the requested window; "
            "forward-filled prices can create zero-return periods and delayed jump returns. Daily close data is strongly recommended.",
            RuntimeWarning,
            stacklevel=2,
        )

    return _filter_and_fill_daily(raw_series, start, end)


def load_local_market_cap_series(ticker: str, start: str, end: str | None, local_market_caps_dir: str | Path) -> pd.Series | None:
    path = _data_file_path(ticker, local_market_caps_dir, "market_cap")
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if data.empty:
        return None
    column_lookup = {column.lower().strip(): column for column in data.columns}
    cap_column = None
    for candidate in ("market_cap", "marketcap", "market cap", "mcap", "cap"):
        if candidate in column_lookup:
            cap_column = column_lookup[candidate]
            break
    if cap_column is None:
        return None
    date_column = _date_column(data)
    series = pd.Series(
        pd.to_numeric(data[cap_column], errors="coerce").to_numpy(),
        index=pd.to_datetime(data[date_column], errors="coerce"),
        name=ticker,
    )
    series = series[series.index.notna()].sort_index().dropna()
    if series.empty:
        return None
    return _filter_and_fill_daily(series, start, end)


def load_local_shares_outstanding(ticker: str, local_market_caps_dir: str | Path) -> float | None:
    path = _data_file_path(ticker, local_market_caps_dir, "market_cap")
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if data.empty:
        return None
    column_lookup = {column.lower().strip(): column for column in data.columns}
    shares_column = None
    for candidate in ("shares_outstanding", "shares", "current_shares_outstanding"):
        if candidate in column_lookup:
            shares_column = column_lookup[candidate]
            break
    if shares_column is None:
        return None
    shares = pd.to_numeric(data[shares_column], errors="coerce").dropna()
    return float(shares.iloc[-1]) if not shares.empty else None


def load_local_market_cap(ticker: str, local_market_caps_dir: str | Path) -> float | None:
    series = load_local_market_cap_series(ticker=ticker, start="1900-01-01", end=None, local_market_caps_dir=local_market_caps_dir)
    return None if series is None or series.empty else float(series.iloc[-1])


def load_member_data_for_membership(
    membership_ranges: pd.DataFrame,
    show_progress: bool = False,
    local_data_dir: str | Path | None = None,
    local_prices_dir: str | Path | None = None,
    local_market_caps_dir: str | Path | None = None,
    report: InputUsageReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load constituent prices plus market-cap priors from local CSV files."""
    required_columns = {"ticker", "start", "end"}
    if not required_columns.issubset(membership_ranges.columns):
        raise ValueError("membership_ranges must contain ticker, start, and end columns")
    if membership_ranges.empty:
        raise ValueError("membership_ranges must not be empty")
    if local_data_dir is not None:
        base = Path(local_data_dir)
        local_prices_dir = local_prices_dir or (base / "price" if (base / "price").exists() else base)
        local_market_caps_dir = local_market_caps_dir or (base / "market_cap" if (base / "market_cap").exists() else base)
    if local_prices_dir is None:
        raise ValueError("a local price CSV directory is required")
    if local_market_caps_dir is None:
        raise ValueError("a local market-cap CSV directory is required")

    prices_by_ticker: list[pd.Series] = []
    caps_by_ticker: list[pd.Series] = []
    shares_by_ticker: dict[str, float] = {}
    missing_prices: list[str] = []
    missing_caps: list[str] = []
    price_ranges: dict[str, tuple[str, str | None]] = {}
    cap_ranges: dict[str, tuple[str, str | None]] = {}

    rows = list(membership_ranges.itertuples(index=False))
    iterator = tqdm(rows, desc="Loading constituent CSVs", unit="ticker", disable=not show_progress)
    for row in iterator:
        ticker = str(row.ticker)
        start = pd.Timestamp(row.start).strftime("%Y-%m-%d")
        end = (pd.Timestamp(row.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        member_range = _display_range(row.start, row.end)

        price_series = load_local_price_series(ticker, start, end, local_prices_dir)
        if price_series is None:
            missing_prices.append(ticker)
            price_ranges[ticker] = member_range
            continue
        prices_by_ticker.append(price_series.rename(ticker))
        _report_series(report, "prices", "csv", ticker, price_series)

        cap_series = load_local_market_cap_series(ticker, start, end, local_market_caps_dir)
        shares_outstanding = load_local_shares_outstanding(ticker, local_market_caps_dir)
        if cap_series is None and shares_outstanding is not None:
            cap_series = (price_series * shares_outstanding).rename(ticker)
            _report_value(report, "shares_outstanding", "csv", ticker)
        elif shares_outstanding is not None:
            _report_value(report, "shares_outstanding", "csv", ticker)
        shares_by_ticker[ticker] = np.nan if shares_outstanding is None else shares_outstanding

        if cap_series is None:
            missing_caps.append(ticker)
            cap_ranges[ticker] = member_range
            continue
        caps_by_ticker.append(cap_series.rename(ticker))
        _report_series(report, "market_caps", "csv", ticker, cap_series)

    if missing_prices or missing_caps:
        raise MissingInputDataError(price_tickers=missing_prices, market_cap_tickers=missing_caps, price_ranges=price_ranges, market_cap_ranges=cap_ranges)
    if not prices_by_ticker:
        raise ValueError("No constituent price data loaded from CSV files")
    if not caps_by_ticker:
        raise ValueError("No constituent market-cap data loaded from CSV files")

    shares = pd.Series(shares_by_ticker, dtype="float64")
    return pd.concat(prices_by_ticker, axis=1).sort_index(), pd.concat(caps_by_ticker, axis=1).sort_index(), shares


def estimate_market_caps_from_prices(prices: pd.DataFrame, shares_outstanding: pd.Series) -> pd.DataFrame:
    """Estimate daily market caps as prices times shares outstanding."""
    if prices.empty:
        raise ValueError("prices must not be empty")
    if shares_outstanding.empty:
        raise ValueError("shares_outstanding must not be empty")
    shares = shares_outstanding.reindex(prices.columns).astype("float64")
    market_caps = prices.ffill().mul(shares, axis=1)
    if market_caps.dropna(how="all").empty:
        raise ValueError("No estimated market-cap data could be calculated")
    return market_caps


def _sp500_frame_from_close(close: pd.Series) -> pd.DataFrame:
    close = close.sort_index().dropna()
    returns = close.pct_change()
    return pd.DataFrame({"sp500_return": returns, "sp500_index": close})


def load_sp500_index(path: str | Path, start: str, end: str | None = None, report: InputUsageReport | None = None) -> pd.DataFrame:
    """Load S&P 500 price-index close levels from a local CSV file."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"index CSV does not exist: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("index CSV must not be empty")
    column_lookup = {column.lower().strip(): column for column in data.columns}
    date_column = column_lookup.get("date", data.columns[0])
    close_column = column_lookup.get("close") or column_lookup.get("sp500_index") or column_lookup.get("index") or column_lookup.get("level") or data.columns[-1]
    close = pd.Series(pd.to_numeric(data[close_column], errors="coerce").to_numpy(), index=pd.to_datetime(data[date_column], errors="coerce"), name="sp500_index")
    close = close[close.index.notna()].sort_index().dropna()
    close = close[close.index >= pd.Timestamp(start)]
    if end is not None:
        close = close[close.index < pd.Timestamp(end)]
    if close.empty:
        raise ValueError("index CSV has no rows in the requested date range")
    _report_series(report, "index", "csv", path.name, close)
    return _sp500_frame_from_close(close)
