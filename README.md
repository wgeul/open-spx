# open-spx

Open Python tooling for approximate bottom-up replication and attribution of S&P 500 price-index returns from user-provided CSV inputs.

`open-spx` builds a point-in-time membership matrix from historical constituent snapshots, loads constituent close prices and market-cap priors from local CSV files, computes daily return contributions, compares the replication against a user-provided S&P 500 price-index series, and can fit a constrained masked RNN adjustment layer against that index return series.

The inferred weights are not official S&P Dow Jones Indices weights. They are model-implied effective replication weights fitted from the supplied inputs. Because each day provides one aggregate index return but hundreds of constituent weights, fitted weights are not uniquely identified. Treat them as diagnostics and approximations, not recovered official index data.

## What This Does Not Do

`open-spx` does not reproduce the official S&P 500 methodology, official float-adjusted shares, investable weight factors, index divisor, dividends, or all corporate-action treatment. It does not recover official constituent weights. The RNN layer is a fitted, prior-constrained explanation of the observed return series, not an independent source of index data.

Final RNN tracking error is in-sample unless the user implements a holdout or walk-forward split. It should be compared with the prior-weight replication error, but it should not be read as independent evidence that the fitted weights match official S&P weights.

## Example Output

The repository includes an example output set in `data/`. It is intended as a compact fixture for inspecting the shape of the generated CSV files and plot artifacts. The RNN metrics below are in-sample for this example and should be read as illustration, not independent validation of official S&P 500 weights.

![SPX vs replicated SPX](data/spx_vs_replicated_spx.png)

Example tracking metrics from `data/replication_metrics.csv`:

| Metric | Value |
| --- | ---: |
| Mean tracking diff daily | 0.0006% |
| Tracking error daily | 0.0264% |
| Tracking error annualized | 0.4192% |
| Observations | 853 |

Latest rows from `data/replication_vs_sp500.csv`:

| Date | SPX | Replicated SPX | Tracking diff |
| --- | ---: | ---: | ---: |
| 2026-05-22 | 7473.47 | 7507.82 | 0.0087% |
| 2026-05-26 | 7519.12 | 7554.80 | 0.0148% |
| 2026-05-27 | 7520.36 | 7557.08 | 0.0138% |
| 2026-05-28 | 7563.63 | 7602.70 | 0.0283% |
| 2026-05-29 | 7580.06 | 7617.60 | -0.0212% |

Largest cumulative positive contributors from `data/cumulative_top_return_contributors.csv`:

| Ticker | Cumulative contribution |
| --- | ---: |
| NVDA | 10.7079% |
| AAPL | 6.0406% |
| AMZN | 4.6502% |
| MSFT | 4.4345% |
| AVGO | 3.8141% |
| GOOGL | 3.3936% |
| GOOG | 3.1410% |
| META | 2.9343% |

Largest cumulative negative contributors from `data/cumulative_top_return_bleeders.csv`:

| Ticker | Cumulative contribution |
| --- | ---: |
| PFE | -0.3972% |
| MMC | -0.3377% |
| UNH | -0.2707% |
| NKE | -0.1742% |
| MRNA | -0.1295% |
| UPS | -0.1135% |
| BMY | -0.1083% |
| PEP | -0.1069% |

`data/largest_market_cap_difference_case.png` provides a second diagnostic plot for the ticker with the largest average market-cap-equivalent gap between the fitted exposure and prior.

## Installation

```bash
git clone https://github.com/wrageul/open-spx.git
cd open-spx
pip install -r requirements.txt
pip install -e . --no-deps
```

The provided `requirements.txt` installs the CPU-only PyTorch wheel from PyTorch's CPU wheel index. It also installs Matplotlib for CLI plots.

## Quickstart

Prepare local CSV inputs, then run:

```bash
open-spx \
  --start 2024-01-01 \
  --constituents data/constituents.csv \
  --index data/sp500_index.csv \
  --local-data-dir data/inputs \
  --out data/run
```

Run with quieter output for logs or CI:

```bash
open-spx --start 2024-01-01 --quiet
```

You can override the two constituent input folders independently:

```bash
open-spx \
  --start 2024-01-01 \
  --index data/sp500_index.csv \
  --local-prices-dir data/prices \
  --local-market-caps-dir data/market_caps \
  --out data/run
```

## Required CSV Inputs

### S&P 500 Index

Provide a local price-index CSV with a date column and a close/level column:

```csv
Date,Close
2024-01-02,4742.83
2024-01-03,4704.81
```

Accepted value column names include `Close`, `sp500_index`, `index`, or `level`. The CLI computes `sp500_return` from this series.

### Historical Constituents

The constituent file should be long form, with one row per date and ticker:

```csv
date,ticker
2024-01-01,A
2024-01-01,B
2024-01-02,A
2024-01-02,C
```

Snapshot-style CSVs with a date column and a comma-separated ticker list are also accepted and normalized internally.

### Constituent Prices

By default, price files are read from `data/inputs/price/TICKER.csv`. Each file should contain a `Date` column and a `Close` column. A lowercase `price` column is also accepted.

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,101.0,103.0,100.5,102.2,1234567
2024-01-03,102.2,104.1,101.7,103.6,1456789
```

Price files are filtered to each ticker's membership date range. Daily and lower-frequency files are accepted; lower-frequency observations are treated as dated anchors and forward-filled to daily frequency before alignment to index trading dates.

### Market Caps Or Shares Outstanding

By default, market-cap prior files are read from `data/inputs/market_cap/TICKER.csv`. Each file may contain either a market-cap time series or shares outstanding.

Market-cap series example:

```csv
date,market_cap
2024-01-02,12345678900
2024-01-03,12400000000
```

Shares-outstanding example:

```csv
date,shares_outstanding
2024-01-02,123456789
```

If a market-cap series is provided, it is used directly as the prior exposure scale. If only shares outstanding is provided, the CLI builds the market-cap prior as `close_price * shares_outstanding`.

## Ticker Mappings

Ticker aliases can be supplied with `--ticker-mappings`, defaulting to `data/ticker_mappings.csv` when that file exists. The mapping target is the ticker used by your local CSV files for that date range. The file uses year-month-day date ranges:

```csv
source_ticker,target_ticker,start,end
OLD,NEW,2024-02-01,
ABC,XYZ,2024-03-04,2024-03-31
```

The `end` column is inclusive and may be blank for an open-ended mapping. Mappings are applied to the daily membership matrix, so inherited memberships from older snapshots can be mapped correctly inside the requested trading range.

By default the CLI applies a 7-calendar-day grace window to open-ended replacement mappings with `--ticker-mapping-grace-days 7`. Use `--ticker-mapping-grace-days 0` to disable this behavior.

## CLI Verbosity

The CLI prints numbered stage messages and uses `tqdm` progress bars for constituent CSV loading and RNN training. If required local files are missing or incomplete, it finishes the constituent pass, prints missing tickers grouped by input type, and exits before training. Use `--quiet` to disable progress bars and stage messages while preserving final tables and errors.

## Outputs

The CLI writes:

```text
historical_constituents.csv
membership_date_ranges.csv
prices.csv
shares_outstanding.csv
market_caps_prior_timeseries.csv
weights_prior_timeseries.csv
replication_prior_weights.csv
return_contributions_prior_weights.csv
weights_rnn_inferred.csv
market_caps_rnn_inferred_free_float.csv
market_cap_differences_rnn_vs_prior.csv
returns.csv
return_contributions.csv
cumulative_top_return_contributors.csv
cumulative_top_return_bleeders.csv
replication_vs_sp500.csv
replication_metrics.csv
input_usage_report.csv
spx_vs_replicated_spx.png
largest_market_cap_difference_case.png
```

`input_usage_report.csv` summarizes rows loaded from local CSV files by stage.

`replication_vs_sp500.csv` includes `sp500_return`, `sp500_index`, `replicated_return`, `replicated_index`, and `tracking_diff`.

`weights_rnn_inferred.csv` contains close-of-day fitted weights. `return_contributions.csv` is indexed by return date and applies the prior close's fitted weights to that day's constituent returns.

`market_caps_prior_timeseries.csv` is the market-cap prior used for weight construction. `market_cap_differences_rnn_vs_prior.csv` compares model-implied market-cap-equivalent exposure with that prior and is intended as a diagnostic.

## Python API

```python
from openspx import (
    constituent_membership_matrix,
    load_historical_constituents,
    load_member_data_for_membership,
    load_sp500_index,
    membership_date_ranges,
    replicate_index,
    replicate_with_weights,
    tracking_metrics,
    train_masked_weight_rnn,
)

start = "2024-01-01"
constituents = load_historical_constituents("data/constituents.csv")
official = load_sp500_index("data/sp500_index.csv", start=start)
membership = constituent_membership_matrix(constituents, official.index)
ranges = membership_date_ranges(membership)

prices, market_caps, shares_outstanding = load_member_data_for_membership(
    ranges,
    local_data_dir="data/inputs",
)
prices = prices.reindex(index=official.index, columns=membership.columns)
market_caps = market_caps.reindex(index=official.index, columns=membership.columns)
market_caps = market_caps.where(membership, 0.0)

prior_replicated, prior_weights, returns, prior_contributions = replicate_index(
    prices=prices,
    market_caps=market_caps,
    base_index=official["sp500_index"],
)
```

## License

Code is licensed under Apache-2.0. Users are responsible for ensuring they have the rights to use and distribute the CSV inputs and generated outputs they create with this project.

This project is independent and is not affiliated with, endorsed by, or sponsored by S&P Dow Jones Indices, S&P Global, or CME Group.
