import pandas as pd

from openspx import (
    DEFAULT_CONSTITUENTS_URL,
    apply_ticker_mappings,
    apply_ticker_mappings_to_membership,
    constituent_membership_matrix,
    load_historical_constituents,
    load_ticker_mappings,
    membership_date_ranges,
    tickers_for_period,
)


def test_default_constituents_source_points_to_fja05680_sp500():
    assert DEFAULT_CONSTITUENTS_URL.startswith(
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
    )

def test_load_historical_constituents_prefers_long_date_ticker_format(tmp_path):
    source = tmp_path / "constituents.csv"
    source.write_text(
        "date,ticker\n"
        "2024-01-01,A\n"
        "2024-01-01,B\n"
        "2024-01-02,A\n",
        encoding="utf-8",
    )

    constituents = load_historical_constituents(source)

    assert constituents.to_dict("records") == [
        {"date": pd.Timestamp("2024-01-01"), "ticker": "A"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "B"},
        {"date": pd.Timestamp("2024-01-02"), "ticker": "A"},
    ]


def test_load_historical_constituents_normalizes_comma_snapshot_format(tmp_path):
    source = tmp_path / "fja_style.csv"
    source.write_text(
        "date,tickers\n"
        "2024-01-01,\"A,B\"\n"
        "2024-01-02,\"A,C\"\n",
        encoding="utf-8",
    )

    constituents = load_historical_constituents(source)

    assert constituents.to_dict("records") == [
        {"date": pd.Timestamp("2024-01-01"), "ticker": "A"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "B"},
        {"date": pd.Timestamp("2024-01-02"), "ticker": "A"},
        {"date": pd.Timestamp("2024-01-02"), "ticker": "C"},
    ]


def test_constituent_membership_matrix_uses_latest_snapshot():
    constituents = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "ticker": ["A", "B", "A", "C"],
        }
    )

    membership = constituent_membership_matrix(
        constituents,
        pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    assert membership.loc["2024-01-02", "A"]
    assert membership.loc["2024-01-02", "B"]
    assert not membership.loc["2024-01-02", "C"]
    assert membership.loc["2024-01-03", "A"]
    assert not membership.loc["2024-01-03", "B"]
    assert membership.loc["2024-01-03", "C"]


def test_tickers_for_period_includes_prior_snapshot():
    constituents = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"]
            ),
            "ticker": ["A", "B", "A", "C"],
        }
    )

    assert tickers_for_period(constituents, "2024-01-02", "2024-01-03") == [
        "A",
        "B",
        "C",
    ]


def test_membership_date_ranges_uses_active_membership_window():
    membership = pd.DataFrame(
        {
            "A": [True, True, False],
            "B": [False, True, True],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    ranges = membership_date_ranges(membership)

    assert ranges.to_dict("records") == [
        {
            "ticker": "A",
            "start": pd.Timestamp("2024-01-02"),
            "end": pd.Timestamp("2024-01-03"),
        },
        {
            "ticker": "B",
            "start": pd.Timestamp("2024-01-03"),
            "end": pd.Timestamp("2024-01-04"),
        },
    ]


def test_load_ticker_mappings_and_apply_aliases(tmp_path):
    mapping_file = tmp_path / "ticker_mappings.csv"
    mapping_file.write_text(
        "source_ticker,target_ticker,start,end\n"
        "CDAY,DAY,2024-02-01,2026-02-28\n",
        encoding="utf-8",
    )
    constituents = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-02-02"]),
            "ticker": ["CDAY", "CDAY"],
        }
    )

    mappings = load_ticker_mappings(mapping_file)
    mapped = apply_ticker_mappings(constituents, mappings)

    assert mappings[["source_ticker", "target_ticker"]].to_dict("records") == [
        {"source_ticker": "CDAY", "target_ticker": "DAY"}
    ]
    assert mapped.to_dict("records") == [
        {"date": pd.Timestamp("2023-12-29"), "ticker": "CDAY"},
        {"date": pd.Timestamp("2024-02-02"), "ticker": "DAY"},
    ]


def test_apply_ticker_mappings_to_membership_is_date_specific(tmp_path):
    mapping_file = tmp_path / "ticker_mappings.csv"
    mapping_file.write_text(
        "source_ticker,target_ticker,start,end\n"
        "PEAK,HCP,2019-11-05,2024-03-03\n"
        "SBNY,BG,2023-03-15,\n"
        "SIVB,PODD,2023-03-15,\n"
        "RE,EG,2023-07-10,2099-12-31\n"
        "ATVI,MSFT,2023-10-13,2023-10-31\n"
        "CDAY,DAY,2024-02-01,2026-02-28\n"
        "PEAK,DOC,2024-03-04,\n"
        "CLT,CTLT,,\n"
        "CTLT,LII,2024-12-23,\n"
        "PXD,VST,2024-05-08,\n"
        "MRO,TPL,2024-11-26,\n"
        "JNPR,DDOG,2025-07-09,\n"
        "HES,XYZ,2025-07-23,\n"
        "PARA,PSKY,2025-08-07,\n",
        encoding="utf-8",
    )
    membership = pd.DataFrame(
        {
            "PEAK": [True, True, False, False, False, False, False, False, False, False, False, False],
            "CDAY": [False, False, True, True, False, False, False, False, False, False, False, False],
            "CLT": [False, False, False, False, True, False, False, False, False, False, False, False],
            "PXD": [False, False, False, False, False, True, False, False, False, False, False, False],
            "MRO": [False, False, False, False, False, False, True, False, False, False, False, False],
            "JNPR": [False, False, False, False, False, False, False, True, False, False, False, False],
            "HES": [False, False, False, False, False, False, False, False, True, False, False, False],
            "PARA": [False, False, False, False, False, False, False, False, False, True, True, False],
        },
        index=pd.to_datetime([
            "2024-03-01",
            "2024-03-04",
            "2023-12-29",
            "2024-02-02",
            "2024-04-01",
            "2024-05-08",
            "2024-11-26",
            "2025-07-09",
            "2025-07-23",
            "2025-08-06",
            "2025-08-07",
            "2025-08-08",
        ]),
    )

    mapped = apply_ticker_mappings_to_membership(
        membership,
        load_ticker_mappings(mapping_file),
    )

    def active(date: str, ticker: str) -> bool:
        return ticker in mapped.columns and bool(mapped.loc[date, ticker])

    assert active("2024-03-01", "HCP")
    assert not active("2024-03-01", "PEAK")
    assert active("2024-03-04", "DOC")
    assert not active("2024-03-04", "PEAK")
    assert active("2023-12-29", "CDAY")
    assert active("2024-02-02", "DAY")
    assert not active("2024-02-02", "CDAY")
    assert active("2024-04-01", "CTLT")
    assert active("2024-05-08", "VST")
    assert active("2024-11-26", "TPL")
    assert active("2025-07-09", "DDOG")
    assert active("2025-07-23", "XYZ")
    assert active("2025-08-06", "PARA")
    assert active("2025-08-07", "PSKY")
    assert not active("2025-08-07", "PARA")

def test_open_ended_ticker_mapping_grace_days_stretches_replacements(tmp_path):
    mapping_file = tmp_path / "ticker_mappings.csv"
    mapping_file.write_text(
        "source_ticker,target_ticker,start,end\n"
        "CDAY,DAY,2024-02-01,2026-02-28\n"
        "CTLT,LII,2024-12-23,\n"
        "PXD,VST,2024-05-08,\n"
        "MRO,TPL,2024-11-26,\n"
        "JNPR,DDOG,2025-07-09,\n"
        "HES,XYZ,2025-07-23,\n",
        encoding="utf-8",
    )
    membership = pd.DataFrame(
        {
            "CDAY": [True, True, False, False, False, False],
            "CTLT": [False, False, True, False, False, False],
            "PXD": [False, False, False, True, False, False],
            "MRO": [False, False, False, False, True, False],
            "HES": [False, False, False, False, False, True],
        },
        index=pd.to_datetime(
            [
                "2023-12-29",
                "2024-02-02",
                "2024-12-18",
                "2024-05-03",
                "2024-11-23",
                "2025-07-19",
            ]
        ),
    )

    mapped = apply_ticker_mappings_to_membership(
        membership,
        load_ticker_mappings(mapping_file),
        grace_days=7,
    )

    def active(date: str, ticker: str) -> bool:
        return ticker in mapped.columns and bool(mapped.loc[date, ticker])

    assert active("2023-12-29", "CDAY")
    assert not active("2023-12-29", "DAY")
    assert active("2024-02-02", "DAY")
    assert active("2024-12-18", "LII")
    assert not active("2024-12-18", "CTLT")
    assert active("2024-05-03", "VST")
    assert active("2024-11-23", "TPL")
    assert active("2025-07-19", "XYZ")



def test_repository_ticker_mappings_include_known_corporate_actions():
    mappings = load_ticker_mappings("data/ticker_mappings.csv")
    rows = {
        (row.source_ticker, row.target_ticker): (
            None if pd.isna(row.start) else row.start.strftime("%Y-%m-%d"),
            None if pd.isna(row.end) else row.end.strftime("%Y-%m-%d"),
        )
        for row in mappings.itertuples(index=False)
    }

    assert rows[("CDAY", "DAY")] == ("2024-02-01", "2026-02-28")
    assert rows[("ATVI", "MSFT")] == ("2023-10-13", "2023-10-31")
    assert rows[("RE", "EG")] == ("2023-07-10", "2099-12-31")
    assert rows[("SBNY", "BG")] == ("2023-03-15", None)
    assert rows[("SIVB", "PODD")] == ("2023-03-15", None)

