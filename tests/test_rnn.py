import pandas as pd
import torch

from openspx import MaskedWeightRNN, make_full_sequence, replicate_with_weights, train_masked_weight_rnn




def test_masked_weight_rnn_can_be_locked_to_prior_weights():
    model = MaskedWeightRNN(
        n_assets=2,
        hidden_size=4,
        max_adjustment_factor=1.0,
    )
    returns = torch.tensor([[[0.01, 0.02], [0.03, 0.01]]], dtype=torch.float32)
    prior_weights = torch.tensor([[[0.7, 0.3], [0.6, 0.4]]], dtype=torch.float32)
    active_mask = torch.ones_like(prior_weights)

    weights = model(returns, prior_weights, active_mask)

    assert torch.allclose(weights, prior_weights, atol=1e-6)


def test_train_masked_weight_rnn_returns_active_normalized_weights():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.00, 0.01, 0.02, 0.00, 0.01, 0.02, 0.00, 0.01],
            "BBB": [0.00, 0.00, 0.01, 0.02, 0.00, 0.01, 0.02, 0.00],
        },
        index=index,
    )
    index_returns = 0.7 * returns["AAA"] + 0.3 * returns["BBB"]
    prior_weights = pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=index)
    active_mask = pd.DataFrame({"AAA": True, "BBB": True}, index=index)

    _, weights = train_masked_weight_rnn(
        returns=returns,
        index_returns=index_returns,
        prior_weights=prior_weights,
        active_mask=active_mask,
        window=3,
        hidden_size=4,
        epochs=2,
        learning_rate=1e-2,
        l2_prior=0.1,
        seed=1,
    )

    assert weights.shape == (7, 2)
    assert weights.index[0] == pd.Timestamp("2024-01-01")
    assert (weights >= 0).all().all()
    assert weights.sum(axis=1).round(6).eq(1.0).all()


def test_replicate_with_weights_uses_inferred_weights():
    returns = pd.DataFrame(
        {"AAA": [0.01, 0.02], "BBB": [0.00, 0.01]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    weights = pd.DataFrame(
        {"AAA": [0.75, 0.25], "BBB": [0.25, 0.75]},
        index=returns.index,
    )

    replicated, contributions = replicate_with_weights(returns, weights)

    assert replicated.index.tolist() == [pd.Timestamp("2024-01-02")]
    assert round(replicated.loc["2024-01-02", "replicated_return"], 4) == 0.0175
    assert contributions.shape == (1, 2)


def test_replicate_with_weights_aligns_index_to_prior_base_index():
    returns = pd.DataFrame(
        {"AAA": [0.10, 0.20]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    weights = pd.DataFrame(
        {"AAA": [1.0, 1.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    base_index = pd.Series(
        [5000.0, 5500.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    replicated, contributions = replicate_with_weights(
        returns,
        weights,
        base_index=base_index,
    )

    assert replicated.loc["2024-01-02", "replicated_index"] == 5500.0
    assert replicated.loc["2024-01-03", "replicated_index"] == 6600.0
    assert contributions.loc["2024-01-03", "AAA"] == 0.20

def test_train_masked_weight_rnn_labels_weights_with_final_window_date_for_new_entrant():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.00, 0.01, 0.02, 0.01, 0.01, 0.01],
            "NEW": [0.00, 0.00, 0.00, 0.05, 0.02, 0.01],
        },
        index=index,
    )
    prior_weights = pd.DataFrame(
        {
            "AAA": [1.0, 1.0, 1.0, 0.9, 0.9, 0.9],
            "NEW": [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
        },
        index=index,
    )
    active_mask = pd.DataFrame(
        {
            "AAA": [True, True, True, True, True, True],
            "NEW": [False, False, False, True, True, True],
        },
        index=index,
    )
    index_returns = (returns * prior_weights).sum(axis=1)

    _, weights = train_masked_weight_rnn(
        returns=returns,
        index_returns=index_returns,
        prior_weights=prior_weights,
        active_mask=active_mask,
        window=3,
        hidden_size=4,
        epochs=1,
        learning_rate=1e-2,
        l2_prior=1.0,
        seed=1,
    )

    assert weights.index[0] == pd.Timestamp("2024-01-01")
    assert weights.loc["2024-01-03", "NEW"] == 0.0
    assert weights.loc["2024-01-04", "NEW"] > 0.0

def test_train_masked_weight_rnn_window_mode_keeps_rolling_window_behavior():
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.00, 0.01, 0.02, 0.00, 0.01, 0.02, 0.00, 0.01],
            "BBB": [0.00, 0.00, 0.01, 0.02, 0.00, 0.01, 0.02, 0.00],
        },
        index=index,
    )
    index_returns = 0.7 * returns["AAA"] + 0.3 * returns["BBB"]
    prior_weights = pd.DataFrame({"AAA": 0.5, "BBB": 0.5}, index=index)
    active_mask = pd.DataFrame({"AAA": True, "BBB": True}, index=index)

    _, weights = train_masked_weight_rnn(
        returns=returns,
        index_returns=index_returns,
        prior_weights=prior_weights,
        active_mask=active_mask,
        window=3,
        hidden_size=4,
        epochs=1,
        learning_rate=1e-2,
        l2_prior=0.1,
        sequence_mode="window",
        seed=1,
    )

    assert weights.shape == (5, 2)
    assert weights.index[0] == pd.Timestamp("2024-01-03")


def test_make_full_sequence_uses_single_continuous_history_after_first_return():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame({"AAA": [0.0, 0.1, 0.2, 0.3]}, index=index)
    index_returns = pd.Series([0.0, 0.1, 0.2, 0.3], index=index)
    prior_weights = pd.DataFrame({"AAA": [1.0, 1.0, 1.0, 1.0]}, index=index)
    active_mask = pd.DataFrame({"AAA": [True, True, True, True]}, index=index)

    x_returns, y_returns, x_priors, x_masks, y_index, dates, columns = make_full_sequence(
        returns,
        index_returns,
        prior_weights,
        active_mask,
    )

    assert x_returns.shape == (1, 3, 1)
    assert y_returns.shape == (1, 3, 1)
    assert x_priors.shape == (1, 3, 1)
    assert x_masks.shape == (1, 3, 1)
    assert y_index.shape == (1, 3)
    assert dates == list(index[:-1])
    assert torch.allclose(x_returns[0, :, 0], torch.tensor([0.0, 0.1, 0.2]))
    assert torch.allclose(y_returns[0, :, 0], torch.tensor([0.1, 0.2, 0.3]))
    assert columns.tolist() == ["AAA"]

