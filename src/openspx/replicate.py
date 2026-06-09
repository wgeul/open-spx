from __future__ import annotations

import pandas as pd


def _normalize_static_weights(
    market_caps: pd.Series,
    columns: pd.Index,
) -> pd.Series:
    weights = market_caps.reindex(columns).fillna(0.0)

    if weights.sum() <= 0:
        raise ValueError("market_caps must contain at least one positive value")

    return weights / weights.sum()


def _normalize_time_dependent_weights(
    market_caps: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    weights = market_caps.reindex(index=prices.index, columns=prices.columns)
    weights = weights.ffill().fillna(0.0)
    row_sums = weights.sum(axis=1)

    if (row_sums > 0).sum() == 0:
        raise ValueError("market_caps must contain at least one positive value")

    return weights.div(row_sums.where(row_sums > 0), axis=0).fillna(0.0)


def replicate_index(
    prices: pd.DataFrame,
    market_caps: pd.Series | pd.DataFrame,
    base_value: float = 100.0,
    base_index: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Replicate index returns bottom-up from constituent prices and weights.

    Parameters
    ----------
    prices:
        DataFrame of adjusted prices, indexed by date, columns are tickers.
    market_caps:
        Series of current constituent market caps indexed by ticker, or a
        DataFrame of time-dependent market caps indexed by date with ticker
        columns.
    base_value:
        Initial index value when base_index is not provided.
    base_index:
        Optional published index level series. When supplied, replicated_index
        starts from the previous available base_index level before the first
        replicated return date.

    Returns
    -------
    replicated:
        DataFrame with replicated daily return and replicated index level.
    weights:
        Normalized constituent weights. Static inputs return a Series;
        time-dependent inputs return a DataFrame indexed like prices.
    returns:
        Constituent daily returns.
    contributions:
        Constituent return contributions.
    """
    if prices.empty:
        raise ValueError("prices must not be empty")

    if market_caps.empty:
        raise ValueError("market_caps must not be empty")

    returns = prices.pct_change()

    if isinstance(market_caps, pd.DataFrame):
        weights = _normalize_time_dependent_weights(market_caps, prices)
        contribution_weights = weights.shift(1)
        contributions = returns.mul(contribution_weights, axis=0)
    else:
        weights = _normalize_static_weights(market_caps, prices.columns)
        contributions = returns.mul(weights, axis=1)

    replicated_return = contributions.sum(axis=1)

    if base_index is not None:
        base_index = base_index.sort_index().dropna()
        prior_base = base_index.reindex(base_index.index.union(replicated_return.index)).ffill()
        prior_base = prior_base.shift(1).reindex(replicated_return.index)
        if prior_base.dropna().empty:
            raise ValueError("base_index must contain a value before replicated dates")
        start_value = float(prior_base.dropna().iloc[0])
    else:
        start_value = base_value

    replicated = pd.DataFrame(
        {
            "replicated_return": replicated_return,
            "replicated_index": (1 + replicated_return.fillna(0)).cumprod()
            * start_value,
        }
    )

    return replicated, weights, returns, contributions


def tracking_metrics(
    comparison: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate tracking metrics for replicated returns versus S&P 500 returns."""
    required_columns = {"replicated_return", "sp500_return"}
    if not required_columns.issubset(comparison.columns):
        raise ValueError("comparison must contain replicated_return and sp500_return")

    tracking_diff = (
        comparison["replicated_return"] - comparison["sp500_return"]
    ).dropna()

    if tracking_diff.empty:
        raise ValueError("comparison must contain at least one tracking difference")

    tracking_error_daily = tracking_diff.std(ddof=1)

    return pd.Series(
        {
            "mean_tracking_diff_daily": tracking_diff.mean(),
            "tracking_error_daily": tracking_error_daily,
            "tracking_error_annualized": tracking_error_daily
            * (periods_per_year ** 0.5),
            "observations": float(tracking_diff.shape[0]),
        },
        dtype="float64",
    )
