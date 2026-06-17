import pandas as pd
import pytest

from openspx import (
    MissingInputDataError,
    estimate_market_caps_from_prices,
    load_local_market_cap,
    load_local_market_cap_series,
    load_local_price_series,
    load_local_shares_outstanding,
    load_member_data_for_membership,
    load_sp500_index,
)


def test_estimate_market_caps_from_prices_returns_time_series():
    prices = pd.DataFrame(
        {"AAA": [50.0, 100.0], "BBB": [100.0, 100.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    shares_outstanding = pd.Series({"AAA": 2.0, "BBB": 1.0})

    market_caps = estimate_market_caps_from_prices(prices, shares_outstanding)

    assert market_caps.shape == prices.shape
    assert market_caps.loc["2024-01-01", "AAA"] == 100.0
    assert market_caps.loc["2024-01-02", "AAA"] == 200.0
    assert market_caps.loc["2024-01-01", "BBB"] == 100.0


def test_load_local_price_series_prefers_close_and_filters_dates(tmp_path):
    local_dir = tmp_path / "prices"
    local_dir.mkdir()
    (local_dir / "DAY.csv").write_text(
        "Date,Close\n"
        "2026-01-01,10.0\n"
        "2026-01-02,11.0\n"
        "2026-01-03,12.0\n",
        encoding="utf-8",
    )

    prices = load_local_price_series("DAY", "2026-01-02", "2026-01-04", local_dir)

    assert prices.to_dict() == {
        pd.Timestamp("2026-01-02"): 11.0,
        pd.Timestamp("2026-01-03"): 12.0,
    }


def test_load_local_price_series_pads_monthly_data_to_daily(tmp_path):
    local_dir = tmp_path / "prices"
    local_dir.mkdir()
    (local_dir / "DAY.csv").write_text(
        "Date,Close\n"
        "2026-01-01,10.0\n"
        "2026-02-01,12.0\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Daily close data is strongly recommended"):
        prices = load_local_price_series("DAY", "2026-01-30", "2026-02-03", local_dir)

    assert prices.to_dict() == {
        pd.Timestamp("2026-01-30"): 10.0,
        pd.Timestamp("2026-01-31"): 10.0,
        pd.Timestamp("2026-02-01"): 12.0,
        pd.Timestamp("2026-02-02"): 12.0,
    }




def test_load_local_price_series_reports_longest_consecutive_business_day_gap(tmp_path):
    local_dir = tmp_path / "prices"
    local_dir.mkdir()
    (local_dir / "DAY.csv").write_text(
        "Date,Close\n"
        "2026-01-02,10.0\n"
        "2026-01-06,11.0\n"
        "2026-01-08,12.0\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="longest consecutive missing business-day run of 1"):
        load_local_price_series("DAY", "2026-01-02", "2026-01-09", local_dir)

def test_load_local_price_series_warns_and_uses_close_when_adj_close_present(tmp_path):
    local_dir = tmp_path / "prices"
    local_dir.mkdir()
    (local_dir / "DAY.csv").write_text(
        "Date,Close,Adj Close\n"
        "2026-01-02,10.0,9.5\n"
        "2026-01-05,11.0,10.4\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="using Close"):
        prices = load_local_price_series("DAY", "2026-01-02", "2026-01-06", local_dir)

    assert prices.loc[pd.Timestamp("2026-01-02")] == 10.0
    assert prices.loc[pd.Timestamp("2026-01-05")] == 11.0


def test_load_local_price_series_warns_when_only_adj_close_is_available(tmp_path):
    local_dir = tmp_path / "prices"
    local_dir.mkdir()
    (local_dir / "DAY.csv").write_text(
        "Date,Adj Close\n"
        "2026-01-02,9.5\n"
        "2026-01-05,10.4\n",
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Adjusted prices"):
        prices = load_local_price_series("DAY", "2026-01-02", "2026-01-06", local_dir)

    assert prices.loc[pd.Timestamp("2026-01-02")] == 9.5
    assert prices.loc[pd.Timestamp("2026-01-05")] == 10.4

def test_load_local_market_cap_series_pads_monthly_data_to_daily(tmp_path):
    input_dir = tmp_path / "inputs"
    market_cap_dir = input_dir / "market_cap"
    market_cap_dir.mkdir(parents=True)
    (market_cap_dir / "DAY.csv").write_text(
        "Date,market_cap\n"
        "2026-01-01,1000\n"
        "2026-02-01,1200\n",
        encoding="utf-8",
    )

    market_caps = load_local_market_cap_series("DAY", "2026-01-30", "2026-02-03", input_dir)

    assert market_caps.to_dict() == {
        pd.Timestamp("2026-01-30"): 1000.0,
        pd.Timestamp("2026-01-31"): 1000.0,
        pd.Timestamp("2026-02-01"): 1200.0,
        pd.Timestamp("2026-02-02"): 1200.0,
    }


def test_local_input_folder_structure_reads_price_market_cap_and_shares(tmp_path):
    input_dir = tmp_path / "inputs"
    price_dir = input_dir / "price"
    market_cap_dir = input_dir / "market_cap"
    price_dir.mkdir(parents=True)
    market_cap_dir.mkdir()
    (price_dir / "DAY.csv").write_text(
        "Date,Close\n2026-01-02,20.0\n2026-01-03,21.0\n",
        encoding="utf-8",
    )
    (market_cap_dir / "DAY.csv").write_text(
        "date,market_cap,shares_outstanding\n2026-01-02,123456789,456\n",
        encoding="utf-8",
    )

    prices = load_local_price_series("DAY", "2026-01-02", "2026-01-04", input_dir)

    assert prices.to_dict() == {
        pd.Timestamp("2026-01-02"): 20.0,
        pd.Timestamp("2026-01-03"): 21.0,
    }
    assert load_local_market_cap("DAY", input_dir) == 123456789.0
    assert load_local_shares_outstanding("DAY", input_dir) == 456.0


def test_load_member_data_for_membership_uses_only_local_csvs(tmp_path):
    input_dir = tmp_path / "inputs"
    price_dir = input_dir / "price"
    market_cap_dir = input_dir / "market_cap"
    price_dir.mkdir(parents=True)
    market_cap_dir.mkdir()
    (price_dir / "AAA.csv").write_text(
        "Date,Close\n2024-01-02,10.0\n2024-01-03,11.0\n",
        encoding="utf-8",
    )
    (market_cap_dir / "AAA.csv").write_text(
        "Date,shares_outstanding\n2024-01-02,100\n",
        encoding="utf-8",
    )
    ranges = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "start": pd.to_datetime(["2024-01-02"]),
            "end": pd.to_datetime(["2024-01-03"]),
        }
    )

    prices, market_caps, shares = load_member_data_for_membership(ranges, local_data_dir=input_dir)

    assert prices["AAA"].to_dict() == {
        pd.Timestamp("2024-01-02"): 10.0,
        pd.Timestamp("2024-01-03"): 11.0,
    }
    assert market_caps["AAA"].to_dict() == {
        pd.Timestamp("2024-01-02"): 1000.0,
        pd.Timestamp("2024-01-03"): 1100.0,
    }
    assert shares["AAA"] == 100.0


def test_load_member_data_reports_missing_csv_inputs(tmp_path):
    input_dir = tmp_path / "inputs"
    (input_dir / "price").mkdir(parents=True)
    (input_dir / "market_cap").mkdir()
    (input_dir / "price" / "NOCAP.csv").write_text(
        "Date,Close\n2024-01-02,10.0\n2024-01-03,11.0\n",
        encoding="utf-8",
    )
    ranges = pd.DataFrame(
        {
            "ticker": ["NOPRICE", "NOCAP"],
            "start": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "end": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        }
    )

    with pytest.raises(MissingInputDataError) as exc_info:
        load_member_data_for_membership(ranges, local_data_dir=input_dir)

    assert exc_info.value.price_tickers == ["NOPRICE"]
    assert exc_info.value.market_cap_tickers == ["NOCAP"]


def test_load_sp500_index_reads_local_csv_and_computes_returns(tmp_path):
    path = tmp_path / "sp500_index.csv"
    path.write_text(
        "Date,Close\n2025-01-02,100.0\n2025-01-03,101.0\n",
        encoding="utf-8",
    )

    official = load_sp500_index(path, "2025-01-01", "2025-01-04")

    assert official.index.tolist() == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert round(official.loc["2025-01-03", "sp500_return"], 4) == 0.01
    assert round(official.loc["2025-01-03", "sp500_index"], 2) == 101.0
