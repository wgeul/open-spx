import pandas as pd

from openspx import replicate_index, tracking_metrics


def test_replicate_index_basic():
    prices = pd.DataFrame(
        {
            "AAA": [100, 110, 121],
            "BBB": [100, 100, 110],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    market_caps = pd.Series(
        {
            "AAA": 75,
            "BBB": 25,
        }
    )

    replicated, weights, returns, contributions = replicate_index(
        prices=prices,
        market_caps=market_caps,
    )

    assert round(weights["AAA"], 2) == 0.75
    assert round(weights["BBB"], 2) == 0.25
    assert "replicated_return" in replicated.columns
    assert "replicated_index" in replicated.columns
    assert returns.shape == prices.shape
    assert contributions.shape == prices.shape


def test_replicate_index_time_dependent_weights_use_prior_period():
    prices = pd.DataFrame(
        {
            "AAA": [100, 110, 121],
            "BBB": [100, 100, 100],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    market_caps = pd.DataFrame(
        {
            "AAA": [50, 80, 20],
            "BBB": [50, 20, 80],
        },
        index=prices.index,
    )

    replicated, weights, returns, contributions = replicate_index(
        prices=prices,
        market_caps=market_caps,
    )

    assert round(weights.loc["2024-01-02", "AAA"], 2) == 0.80
    assert round(weights.loc["2024-01-02", "BBB"], 2) == 0.20
    assert round(contributions.loc["2024-01-02", "AAA"], 4) == 0.05
    assert round(replicated.loc["2024-01-03", "replicated_return"], 4) == 0.08
    assert returns.shape == prices.shape
    assert contributions.shape == prices.shape


def test_tracking_metrics_calculates_annualized_tracking_error():
    comparison = pd.DataFrame(
        {
            "replicated_return": [0.01, 0.02, 0.00],
            "sp500_return": [0.00, 0.01, 0.01],
        }
    )

    metrics = tracking_metrics(comparison, periods_per_year=4)

    assert round(metrics["mean_tracking_diff_daily"], 6) == round(1 / 300, 6)
    assert round(metrics["tracking_error_daily"], 6) == round(0.011547005383792516, 6)
    assert round(metrics["tracking_error_annualized"], 6) == round(0.023094010767585032, 6)
    assert metrics["observations"] == 3.0


def test_replicate_index_aligns_index_to_prior_base_index():
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 132.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    market_caps = pd.Series({"AAA": 1.0})
    base_index = pd.Series(
        [5000.0, 5100.0],
        index=pd.to_datetime(["2023-12-29", "2024-01-01"]),
    )

    replicated, _, _, _ = replicate_index(
        prices=prices,
        market_caps=market_caps,
        base_index=base_index,
    )

    assert replicated.loc["2024-01-01", "replicated_index"] == 5000.0
    assert replicated.loc["2024-01-02", "replicated_index"] == 5500.0
    assert replicated.loc["2024-01-03", "replicated_index"] == 6600.0
