from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from .constituents import (
    DEFAULT_CONSTITUENTS_URL,
    apply_ticker_mappings_to_membership,
    constituent_membership_matrix,
    load_historical_constituents,
    load_ticker_mappings,
    membership_date_ranges,
)
from .replicate import replicate_index, tracking_metrics
from .rnn import replicate_with_weights, train_masked_weight_rnn
from .data import (
    InputUsageReport,
    MissingInputDataError,
    format_input_usage_report,
    load_member_data_for_membership,
    load_sp500_index,
)





def format_missing_ticker_lines(
    tickers: list[str],
    ranges: dict[str, tuple[str, str | None]] | None = None,
) -> str:
    ranges = ranges or {}
    lines = []
    for ticker in tickers:
        suffix = ""
        if ticker in ranges:
            start, end = ranges[ticker]
            suffix = f" ({start} to {end or 'open-ended'})"
        lines.append(f"  - {ticker}{suffix}")
    return "\n".join(lines)

def write_comparison_plot(comparison, path: Path) -> None:
    """Write a time-series plot comparing published SPX and replicated SPX."""
    plot_data = comparison[["sp500_index", "replicated_index"]].dropna()
    if plot_data.empty:
        raise ValueError("comparison must contain non-empty index series to plot")

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(
        plot_data.index,
        plot_data["sp500_index"],
        label="SPX (^SPX)",
        color="#1f77b4",
        linewidth=1.8,
    )
    ax.plot(
        plot_data.index,
        plot_data["replicated_index"],
        label="Replicated SPX",
        color="#d62728",
        linewidth=1.6,
        alpha=0.9,
    )
    ax.set_title("SPX vs Replicated SPX")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index level")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ticker_with_largest_average_market_cap_difference(market_cap_equivalent_exposure_gap):
    """Return ticker with largest mean absolute market-cap difference."""
    numeric_differences = market_cap_equivalent_exposure_gap.apply(pd.to_numeric, errors="coerce")
    mean_abs_difference = numeric_differences.abs().mean(axis=0).dropna()
    if mean_abs_difference.empty:
        raise ValueError("market_cap_equivalent_exposure_gap must contain at least one non-null value")
    return mean_abs_difference.idxmax()


def write_market_cap_difference_plot(
    prior_market_caps,
    inferred_market_caps,
    market_cap_equivalent_exposure_gap,
    path: Path,
    prior_label: str,
) -> str:
    """Plot prior vs inferred market-cap-equivalent exposure for largest gap."""
    ticker = ticker_with_largest_average_market_cap_difference(market_cap_equivalent_exposure_gap)
    plot_data = (
        prior_market_caps[[ticker]]
        .rename(columns={ticker: prior_label})
        .join(
            inferred_market_caps[[ticker]].rename(
                columns={ticker: "Model-Implied Effective Exposure"}
            ),
            how="outer",
        )
        .dropna()
    )
    if plot_data.empty:
        raise ValueError(f"No overlapping market-cap data to plot for {ticker}")

    prior = plot_data[prior_label]
    inferred = plot_data["Model-Implied Effective Exposure"]
    mean_difference = (inferred - prior).mean()
    mean_abs_pct = ((inferred - prior).abs() / prior.replace(0.0, float("nan"))).mean()

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(
        plot_data.index,
        prior,
        label=prior_label,
        color="#1f77b4",
        linewidth=1.8,
    )
    ax.plot(
        plot_data.index,
        inferred,
        label="Model-Implied Effective Exposure",
        color="#d62728",
        linewidth=1.8,
    )
    ax.fill_between(
        plot_data.index,
        prior.to_numpy(),
        inferred.to_numpy(),
        color="#ff9896",
        alpha=0.22,
        label="Difference",
    )
    ax.set_title(f"Largest Average Market-Cap Difference: {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Market cap / exposure scale")
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "Inferred series is model-implied exposure, not observed free-float market cap.\n"
        f"Mean difference: {mean_difference:,.0f}; mean absolute percent gap: {mean_abs_pct:.2%}",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    fig.autofmt_xdate()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(ticker)


def cumulative_top_contributors(contributions, top_n: int = 25):
    """Return cumulative return-date contributions for the most important names."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    cumulative = contributions.fillna(0.0).cumsum()
    if cumulative.empty:
        raise ValueError("contributions must not be empty")

    importance = cumulative.iloc[-1].abs().sort_values(ascending=False)
    top_tickers = importance.head(top_n).index.tolist()

    return cumulative_contributor_table(contributions, top_tickers)


def cumulative_top_bleeders(contributions, top_n: int = 25):
    """Return cumulative return-date contributions for the largest detractors."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    cumulative = contributions.fillna(0.0).cumsum()
    if cumulative.empty:
        raise ValueError("contributions must not be empty")

    bleeders = cumulative.iloc[-1].sort_values(ascending=True)
    top_tickers = bleeders.head(top_n).index.tolist()

    return cumulative_contributor_table(contributions, top_tickers)


def cumulative_contributor_table(contributions, tickers):
    cumulative = contributions.fillna(0.0).cumsum()
    result = cumulative.loc[:, tickers].copy()
    result.insert(
        0,
        "Total Replicated Return Contribution",
        contributions.fillna(0.0).sum(axis=1).cumsum(),
    )
    return result

def tracking_metrics_by_model(prior_comparison, fitted_comparison):
    rows = []
    for model, comparison, fitted_layer in (
        ("prior_market_cap_weights", prior_comparison, "no"),
        ("rnn_model_implied_weights", fitted_comparison, "yes_ex_post_in_sample"),
    ):
        metrics = tracking_metrics(comparison)
        rows.append(
            {
                "model": model,
                "daily_tracking_error": metrics.get("tracking_error_daily"),
                "annualized_tracking_error": metrics.get("tracking_error_annualized"),
                "mean_tracking_diff_daily": metrics.get("mean_tracking_diff_daily"),
                "observations": metrics.get("observations"),
                "fitted_layer": fitted_layer,
            }
        )
    return pd.DataFrame(rows)


def build_anomaly_report(
    returns,
    membership,
    exposure_gap_pct,
    large_return_threshold: float = 0.15,
    exposure_gap_threshold: float = 25.0,
):
    records = []

    large_returns = returns.stack().dropna()
    large_returns = large_returns[large_returns.abs() >= large_return_threshold]
    for (date, ticker), value in large_returns.items():
        records.append(
            {
                "Date": date,
                "Ticker": ticker,
                "anomaly_type": "large_single_name_return",
                "value": float(value),
                "note": f"absolute return >= {large_return_threshold:.0%}",
            }
        )

    membership_changes = membership.astype(bool).astype(int).diff().fillna(0)
    transitions = membership_changes.stack()
    transitions = transitions[transitions != 0]
    for (date, ticker), value in transitions.items():
        records.append(
            {
                "Date": date,
                "Ticker": ticker,
                "anomaly_type": "membership_transition",
                "value": int(value),
                "note": "entered" if value > 0 else "exited",
            }
        )

    gaps = exposure_gap_pct.stack().dropna()
    gaps = gaps[gaps.abs() >= exposure_gap_threshold]
    for (date, ticker), value in gaps.items():
        records.append(
            {
                "Date": date,
                "Ticker": ticker,
                "anomaly_type": "model_vs_prior_exposure_gap_pct",
                "value": float(value),
                "note": f"absolute fitted-vs-prior gap >= {exposure_gap_threshold:.1f}%",
            }
        )

    columns = ["Date", "Ticker", "anomaly_type", "value", "note"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values(
        ["Date", "Ticker", "anomaly_type"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open S&P 500 bottom-up replication using user-provided CSV data."
    )

    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data")
    parser.add_argument(
        "--constituents",
        default=DEFAULT_CONSTITUENTS_URL,
        help=(
            "Path or URL to historical constituents in long date,ticker format. "
            "The default uses fja05680/sp500 and normalizes it to that format."
        ),
    )
    parser.add_argument(
        "--ticker-mappings",
        default="data/ticker_mappings.csv",
        help=(
            "Optional CSV of date-ranged ticker aliases with "
            "source_ticker,target_ticker,start,end columns. Used for renamed "
            "symbols in historical constituent data."
        ),
    )
    parser.add_argument(
        "--ticker-mapping-grace-days",
        type=int,
        default=7,
        help=(
            "Stretch open-ended ticker replacement mappings backward by this "
            "many calendar days. Closed mapping ranges keep their exact dates."
        ),
    )
    parser.add_argument(
        "--rnn-sequence-mode",
        choices=["full", "window"],
        default="full",
        help=(
            "RNN training sequence construction. 'full' fits one continuous "
            "in-sample history; 'window' uses overlapping rolling windows."
        ),
    )
    parser.add_argument(
        "--rnn-window",
        type=int,
        default=60,
        help="Rolling sequence length used only with --rnn-sequence-mode window.",
    )
    parser.add_argument("--rnn-hidden-size", type=int, default=64)
    parser.add_argument("--rnn-epochs", type=int, default=200)
    parser.add_argument("--rnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rnn-l2-prior", type=float, default=25.0)
    parser.add_argument("--rnn-l2-smoothness", type=float, default=10.0)
    parser.add_argument(
        "--rnn-max-adjustment-factor",
        type=float,
        default=1.5,
        help=(
            "Maximum per-window multiplicative adjustment factor applied around "
            "prior weights before normalization. Use 1.0 to disable RNN "
            "deviations from the prior."
        ),
    )
    parser.add_argument(
        "--top-contributors",
        type=int,
        default=25,
        help=(
            "Number of top absolute cumulative return contributors to include "
            "in cumulative_top_return_contributors.csv."
        ),
    )
    parser.add_argument(
        "--top-bleeders",
        type=int,
        default=25,
        help=(
            "Number of most negative cumulative return contributors to include "
            "in cumulative_top_return_bleeders.csv."
        ),
    )
    parser.add_argument(
        "--index",
        default="data/sp500_index.csv",
        help="Path to a local S&P 500 price-index CSV with Date and Close columns.",
    )
    parser.add_argument(
        "--local-data-dir",
        default="data/inputs",
        help=(
            "Directory containing local CSV inputs. Price files are read from "
            "DATA_DIR/price/TICKER.csv; market-cap or shares files are read "
            "from DATA_DIR/market_cap/TICKER.csv."
        ),
    )
    parser.add_argument(
        "--local-prices-dir",
        default=None,
        help="Optional override for local price CSVs.",
    )
    parser.add_argument(
        "--local-market-caps-dir",
        default=None,
        help="Optional override for local market-cap or shares-outstanding CSVs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars and stage messages.",
    )

    args = parser.parse_args()

    show_progress = not args.quiet
    input_usage_report = InputUsageReport()

    def log(message: str) -> None:
        if show_progress:
            tqdm.write(message)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("[1/7] Loading historical constituents")
    constituents = load_historical_constituents(args.constituents)
    ticker_mappings = load_ticker_mappings(args.ticker_mappings)
    if not ticker_mappings.empty:
        log(f"      Loaded {len(ticker_mappings)} ticker mapping(s) from {args.ticker_mappings}")
    log("[2/7] Loading S&P 500 index returns from CSV")
    official = load_sp500_index(
        args.index,
        args.start,
        args.end,
        report=input_usage_report,
    )
    log("[3/7] Building point-in-time membership matrix")
    membership = constituent_membership_matrix(constituents, official.index)
    if not ticker_mappings.empty:
        membership = apply_ticker_mappings_to_membership(
            membership,
            ticker_mappings,
            grace_days=args.ticker_mapping_grace_days,
        )
    ranges = membership_date_ranges(membership)

    local_data_dir = Path(args.local_data_dir) if args.local_data_dir else None
    local_prices_dir = Path(args.local_prices_dir) if args.local_prices_dir else None
    local_market_caps_dir = (
        Path(args.local_market_caps_dir) if args.local_market_caps_dir else None
    )

    log(f"[4/7] Loading constituent prices and market-cap priors from CSV for {len(ranges)} tickers")
    if local_data_dir is not None:
        log(f"      Reading local input files under {local_data_dir}")
    if local_prices_dir is not None:
        log(f"      Reading prices from {local_prices_dir}")
    if local_market_caps_dir is not None:
        log(f"      Reading market caps or shares outstanding from {local_market_caps_dir}")

    try:
        prices, market_caps, shares_outstanding = load_member_data_for_membership(
            ranges,
            show_progress=show_progress,
            local_data_dir=local_data_dir,
            local_prices_dir=local_prices_dir,
            local_market_caps_dir=local_market_caps_dir,
            report=input_usage_report,
        )
        prices = prices.reindex(index=official.index, columns=membership.columns)
        market_caps = market_caps.reindex(index=official.index, columns=membership.columns)
        market_caps = market_caps.where(membership, 0.0)
        log("[5/7] Building prior weight time series")
    except MissingInputDataError as exc:
        sections = []
        if exc.price_tickers:
            failed_prices = format_missing_ticker_lines(exc.price_tickers, exc.price_ranges)
            sections.append("Missing price data:\n" + failed_prices)
        if exc.market_cap_tickers:
            failed_caps = format_missing_ticker_lines(
                exc.market_cap_tickers,
                exc.market_cap_ranges,
            )
            sections.append("Missing market-cap or shares-outstanding data:\n" + failed_caps)
        parser.exit(
            1,
            "Required constituent CSV inputs are missing or incomplete:\n"
            + "\n\n".join(sections)
            + "\n\nAdd price CSVs under data/inputs/price and market-cap or "
            "shares-outstanding CSVs under data/inputs/market_cap, update "
            "symbol mappings, or choose a period covered by your local inputs.\n",
        )
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")

    log("[6/7] Replicating from prior weights")
    prior_replicated, prior_weights, returns, prior_contributions = replicate_index(
        prices=prices,
        market_caps=market_caps,
        base_index=official["sp500_index"],
    )

    log(f"[7/7] Training masked RNN on SPX returns ({args.rnn_sequence_mode} sequence) and computing final tracking error")
    _, weights = train_masked_weight_rnn(
        returns=returns,
        index_returns=official["sp500_return"],
        prior_weights=prior_weights,
        active_mask=membership,
        window=args.rnn_window,
        hidden_size=args.rnn_hidden_size,
        epochs=args.rnn_epochs,
        learning_rate=args.rnn_learning_rate,
        l2_prior=args.rnn_l2_prior,
        l2_smoothness=args.rnn_l2_smoothness,
        max_adjustment_factor=args.rnn_max_adjustment_factor,
        sequence_mode=args.rnn_sequence_mode,
        show_progress=show_progress,
    )
    replicated, contributions = replicate_with_weights(
        returns,
        weights,
        base_index=official["sp500_index"],
    )

    total_market_cap = market_caps.sum(axis=1).replace(0.0, float("nan"))
    active_weights_mask = membership.reindex(
        index=weights.index,
        columns=weights.columns,
        fill_value=False,
    ).astype(bool)
    model_implied_effective_exposures = weights.mul(total_market_cap, axis=0).where(
        active_weights_mask
    )
    prior_market_caps_for_weights = market_caps.reindex(
        index=weights.index,
        columns=weights.columns,
    ).where(active_weights_mask)
    market_cap_equivalent_exposure_gap = (
        model_implied_effective_exposures - prior_market_caps_for_weights
    )
    market_cap_equivalent_exposure_gap_pct = (
        market_cap_equivalent_exposure_gap
        .div(prior_market_caps_for_weights.replace(0.0, float("nan")))
        .mul(100.0)
    )

    def long_market_cap_series(frame, name):
        stacked = frame.stack().dropna().rename(name)
        stacked.index = stacked.index.set_names(["Date", "Ticker"])
        return stacked

    market_cap_prior_label = "Market Cap Prior"
    market_cap_equivalent_exposure_gap_pct_label = "Exposure Gap (%) of Prior"

    market_cap_difference_table = (
        long_market_cap_series(
            prior_market_caps_for_weights,
            market_cap_prior_label,
        )
        .to_frame()
        .join(
            long_market_cap_series(
                model_implied_effective_exposures,
                "Model-Implied Effective Exposure",
            )
        )
        .join(
            long_market_cap_series(
                market_cap_equivalent_exposure_gap,
                "Exposure Gap",
            )
        )
        .join(
            long_market_cap_series(
                market_cap_equivalent_exposure_gap_pct,
                market_cap_equivalent_exposure_gap_pct_label,
            )
        )
        .reset_index()
    )

    cumulative_contributors = cumulative_top_contributors(
        contributions,
        top_n=args.top_contributors,
    )
    cumulative_bleeders = cumulative_top_bleeders(
        contributions,
        top_n=args.top_bleeders,
    )

    comparison = replicated.join(official, how="inner")
    comparison["tracking_diff"] = (
        comparison["replicated_return"] - comparison["sp500_return"]
    )
    metrics = tracking_metrics(comparison)
    prior_comparison = prior_replicated.join(official, how="inner")
    prior_comparison["tracking_diff"] = (
        prior_comparison["replicated_return"] - prior_comparison["sp500_return"]
    )
    metrics_by_model = tracking_metrics_by_model(prior_comparison, comparison)
    anomaly_report = build_anomaly_report(
        returns,
        membership.reindex(index=returns.index, columns=returns.columns, fill_value=False),
        market_cap_equivalent_exposure_gap_pct,
    )

    constituents.to_csv(out_dir / "historical_constituents.csv", index=False)
    ranges.to_csv(out_dir / "membership_date_ranges.csv", index=False)
    prices.to_csv(out_dir / "prices.csv")
    if shares_outstanding is not None and not shares_outstanding.dropna().empty:
        shares_outstanding.to_csv(out_dir / "shares_outstanding.csv")
    market_caps.to_csv(out_dir / "market_caps_prior_timeseries.csv")
    prior_weights.to_csv(out_dir / "weights_prior_timeseries.csv")
    prior_replicated.to_csv(out_dir / "replication_prior_weights.csv")
    prior_contributions.to_csv(out_dir / "return_contributions_prior_weights.csv")
    weights.to_csv(out_dir / "weights_model_implied.csv")
    model_implied_effective_exposures.to_csv(
        out_dir / "effective_exposures_model_fit.csv"
    )
    market_cap_difference_table.to_csv(
        out_dir / "market_cap_equivalent_exposure_gap.csv",
        index=False,
    )
    returns.to_csv(out_dir / "returns.csv")
    contributions.to_csv(out_dir / "return_contributions.csv")
    cumulative_contributors.to_csv(
        out_dir / "cumulative_top_return_contributors.csv"
    )
    cumulative_bleeders.to_csv(
        out_dir / "cumulative_top_return_bleeders.csv"
    )
    comparison.to_csv(out_dir / "replication_vs_sp500.csv")
    metrics.to_csv(out_dir / "replication_metrics.csv", header=["value"])
    metrics_by_model.to_csv(out_dir / "replication_metrics_by_model.csv", index=False)
    anomaly_report.to_csv(out_dir / "anomaly_report.csv", index=False)
    input_usage_report.summary().to_csv(out_dir / "input_usage_report.csv", index=False)
    plot_path = out_dir / "spx_vs_replicated_spx.png"
    write_comparison_plot(comparison, plot_path)
    market_cap_plot_path = out_dir / "largest_market_cap_difference_case.png"
    market_cap_plot_ticker = write_market_cap_difference_plot(
        prior_market_caps_for_weights,
        model_implied_effective_exposures,
        market_cap_equivalent_exposure_gap,
        market_cap_plot_path,
        market_cap_prior_label,
    )

    if show_progress:
        tqdm.write("Latest comparison rows:")
        tqdm.write(comparison.tail().to_string())
        tqdm.write("Replication metrics:")
        tqdm.write(metrics.to_frame("value").to_string())
        tqdm.write("Input usage report:")
        tqdm.write(format_input_usage_report(input_usage_report))
        tqdm.write(f"Comparison plot written to {plot_path}")
        tqdm.write(
            "Largest market-cap difference plot written to "
            f"{market_cap_plot_path} for {market_cap_plot_ticker}"
        )
    else:
        print(comparison.tail())
        print(metrics.to_frame("value"))
        print(format_input_usage_report(input_usage_report))


if __name__ == "__main__":
    main()
