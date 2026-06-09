import pandas as pd

from openspx import replicate_with_weights
from openspx.cli import (
    cumulative_top_bleeders,
    cumulative_top_contributors,
    format_missing_ticker_lines,
    ticker_with_largest_average_market_cap_difference,
    write_market_cap_difference_plot,
)


def test_cumulative_top_contributors_ranks_by_absolute_final_contribution():
    contributions = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01],
            "BBB": [-0.03, 0.00, 0.00],
            "CCC": [0.005, 0.005, 0.005],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    result = cumulative_top_contributors(contributions, top_n=2)

    assert result.columns.tolist() == [
        "Total Replicated Return Contribution",
        "BBB",
        "AAA",
    ]
    assert round(result.loc["2024-01-03", "Total Replicated Return Contribution"], 6) == 0.005
    assert result.loc["2024-01-03", "BBB"] == -0.03
    assert round(result.loc["2024-01-03", "AAA"], 6) == 0.02


def test_cumulative_top_bleeders_ranks_by_most_negative_final_contribution():
    contributions = pd.DataFrame(
        {
            "AAA": [0.01, -0.04, 0.00],
            "BBB": [-0.01, -0.01, -0.02],
            "CCC": [0.02, 0.01, -0.01],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    result = cumulative_top_bleeders(contributions, top_n=2)

    assert result.columns.tolist() == [
        "Total Replicated Return Contribution",
        "BBB",
        "AAA",
    ]
    assert round(result.loc["2024-01-03", "BBB"], 6) == -0.04
    assert round(result.loc["2024-01-03", "AAA"], 6) == -0.03


def test_cumulative_tables_use_lagged_rnn_contributions():
    returns = pd.DataFrame(
        {
            "AAA": [0.10, 0.20, -0.05],
            "BBB": [0.00, -0.10, 0.02],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    weights = pd.DataFrame(
        {
            "AAA": [0.75, 0.25, 0.25],
            "BBB": [0.25, 0.75, 0.75],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    replicated, contributions = replicate_with_weights(returns, weights)
    contributors = cumulative_top_contributors(contributions, top_n=2)
    bleeders = cumulative_top_bleeders(contributions, top_n=2)

    expected_returns = pd.Series(
        [0.075, -0.025, 0.0025],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        name="replicated_return",
    )
    pd.testing.assert_series_equal(replicated["replicated_return"], expected_returns)
    pd.testing.assert_series_equal(
        contributors["Total Replicated Return Contribution"],
        expected_returns.cumsum().rename("Total Replicated Return Contribution"),
    )
    pd.testing.assert_series_equal(
        bleeders["Total Replicated Return Contribution"],
        expected_returns.cumsum().rename("Total Replicated Return Contribution"),
    )

def test_format_missing_ticker_lines_includes_ranges():
    assert format_missing_ticker_lines(
        ["AAA", "BBB"],
        {"AAA": ("2024-01-01", "2024-01-31"), "BBB": ("2024-02-01", None)},
    ) == "  - AAA (2024-01-01 to 2024-01-31)\n  - BBB (2024-02-01 to open-ended)"

def test_ticker_with_largest_average_market_cap_difference_uses_absolute_mean():
    differences = pd.DataFrame(
        {
            "AAA": [100.0, -100.0, 100.0],
            "BBB": [-500.0, 10.0, 10.0],
            "CCC": [None, None, None],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    assert ticker_with_largest_average_market_cap_difference(differences) == "BBB"

def test_write_market_cap_difference_plot_writes_png(tmp_path):
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prior = pd.DataFrame(
        {"AAA": [100.0, 100.0, 100.0], "BBB": [200.0, 200.0, 200.0]},
        index=index,
    )
    inferred = pd.DataFrame(
        {"AAA": [105.0, 106.0, 107.0], "BBB": [250.0, 260.0, 270.0]},
        index=index,
    )
    differences = inferred - prior
    path = tmp_path / "market_cap_case.png"

    ticker = write_market_cap_difference_plot(
        prior,
        inferred,
        differences,
        path,
        "Market Cap Prior",
    )

    assert ticker == "BBB"
    assert path.exists()
    assert path.stat().st_size > 0

