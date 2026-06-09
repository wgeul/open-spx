from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm


class MaskedWeightRNN(nn.Module):
    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        max_adjustment_factor: float = 1.5,
    ) -> None:
        super().__init__()

        if max_adjustment_factor < 1.0:
            raise ValueError("max_adjustment_factor must be >= 1.0")

        self.max_log_adjustment = float(np.log(max_adjustment_factor))
        self.rnn = nn.GRU(
            input_size=n_assets * 3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_assets)

    def forward(
        self,
        returns: torch.Tensor,
        prior_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        returns:       [batch, time, assets]
        prior_weights: [batch, time, assets]
        active_mask:   [batch, time, assets]

        returns:
        weights:       [batch, time, assets]
        """
        active_mask = active_mask.float()

        x = torch.cat(
            [
                returns * active_mask,
                prior_weights * active_mask,
                active_mask,
            ],
            dim=-1,
        )

        h, _ = self.rnn(x)
        raw_adjustments = self.head(h)
        bounded_adjustments = torch.tanh(raw_adjustments) * self.max_log_adjustment

        prior_logits = torch.log(prior_weights.clamp_min(1e-8))
        masked_logits = prior_logits + bounded_adjustments
        masked_logits = masked_logits.masked_fill(active_mask <= 0, -1e9)

        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * active_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return weights


def _aligned_rnn_inputs(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    prior_weights: pd.DataFrame,
    active_mask: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Index]:
    common_index = (
        returns.index.intersection(index_returns.index)
        .intersection(prior_weights.index)
        .intersection(active_mask.index)
    )

    returns = returns.loc[common_index].fillna(0.0)
    index_returns = index_returns.loc[common_index].fillna(0.0)
    prior_weights = prior_weights.loc[common_index].fillna(0.0)
    active_mask = active_mask.loc[common_index].astype(bool)

    common_columns = returns.columns.intersection(prior_weights.columns).intersection(
        active_mask.columns
    )

    returns = returns[common_columns]
    prior_weights = prior_weights[common_columns]
    active_mask = active_mask[common_columns]

    active_rows = active_mask.any(axis=1)
    returns = returns.loc[active_rows]
    index_returns = index_returns.loc[active_rows]
    prior_weights = prior_weights.loc[active_rows]
    active_mask = active_mask.loc[active_rows]

    prior_weights = prior_weights.where(active_mask, 0.0)
    prior_weights = prior_weights.div(
        prior_weights.sum(axis=1).replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)

    return returns, index_returns, prior_weights, active_mask, common_columns


def make_masked_sequences(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    prior_weights: pd.DataFrame,
    active_mask: pd.DataFrame,
    window: int = 60,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[pd.Timestamp],
    pd.Index,
]:
    returns, index_returns, prior_weights, active_mask, common_columns = (
        _aligned_rnn_inputs(returns, index_returns, prior_weights, active_mask)
    )
    common_index = returns.index

    x_returns = []
    y_returns = []
    x_priors = []
    x_masks = []
    y_index = []
    dates = []

    for i in range(window, len(common_index)):
        r = returns.iloc[i - window : i]
        target_r = returns.iloc[i - window + 1 : i + 1]
        p = prior_weights.iloc[i - window : i]
        m = active_mask.iloc[i - window : i]

        if not m.any(axis=1).all():
            continue

        x_returns.append(r.to_numpy(dtype=np.float32))
        y_returns.append(target_r.to_numpy(dtype=np.float32))
        x_priors.append(p.to_numpy(dtype=np.float32))
        x_masks.append(m.to_numpy(dtype=np.float32))
        y_index.append(index_returns.iloc[i - window + 1 : i + 1].to_numpy(dtype=np.float32))
        dates.append(common_index[i - 1])

    if not x_returns:
        raise ValueError("No valid training sequences created")

    return (
        torch.tensor(np.stack(x_returns)),
        torch.tensor(np.stack(y_returns)),
        torch.tensor(np.stack(x_priors)),
        torch.tensor(np.stack(x_masks)),
        torch.tensor(np.stack(y_index)),
        dates,
        common_columns,
    )


def make_full_sequence(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    prior_weights: pd.DataFrame,
    active_mask: pd.DataFrame,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[pd.Timestamp],
    pd.Index,
]:
    returns, index_returns, prior_weights, active_mask, common_columns = (
        _aligned_rnn_inputs(returns, index_returns, prior_weights, active_mask)
    )

    if len(returns.index) < 2:
        raise ValueError("No valid full-sequence training data created")

    feature_returns = returns.iloc[:-1]
    target_returns = returns.iloc[1:]
    target_index_returns = index_returns.iloc[1:]
    prior_weights = prior_weights.iloc[:-1]
    active_mask = active_mask.iloc[:-1]

    return (
        torch.tensor(feature_returns.to_numpy(dtype=np.float32))[None, :, :],
        torch.tensor(target_returns.to_numpy(dtype=np.float32))[None, :, :],
        torch.tensor(prior_weights.to_numpy(dtype=np.float32))[None, :, :],
        torch.tensor(active_mask.to_numpy(dtype=np.float32))[None, :, :],
        torch.tensor(target_index_returns.to_numpy(dtype=np.float32))[None, :],
        list(feature_returns.index),
        common_columns,
    )


def train_masked_weight_rnn(
    returns: pd.DataFrame,
    index_returns: pd.Series,
    prior_weights: pd.DataFrame,
    active_mask: pd.DataFrame,
    window: int = 60,
    hidden_size: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    l2_prior: float = 25.0,
    l2_smoothness: float = 10.0,
    max_adjustment_factor: float = 1.5,
    sequence_mode: str = "full",
    seed: int | None = 0,
    show_progress: bool = False,
) -> tuple[MaskedWeightRNN, pd.DataFrame]:
    if seed is not None:
        torch.manual_seed(seed)

    sequence_mode = sequence_mode.lower()
    if sequence_mode == "full":
        x_returns, y_returns, x_priors, x_masks, y_index, dates, columns = make_full_sequence(
            returns=returns,
            index_returns=index_returns,
            prior_weights=prior_weights,
            active_mask=active_mask,
        )
    elif sequence_mode == "window":
        x_returns, y_returns, x_priors, x_masks, y_index, dates, columns = make_masked_sequences(
            returns=returns,
            index_returns=index_returns,
            prior_weights=prior_weights,
            active_mask=active_mask,
            window=window,
        )
    else:
        raise ValueError("sequence_mode must be 'full' or 'window'")

    n_assets = x_returns.shape[-1]

    model = MaskedWeightRNN(
        n_assets=n_assets,
        hidden_size=hidden_size,
        max_adjustment_factor=max_adjustment_factor,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_iter = tqdm(
        range(epochs),
        desc="Training masked weight RNN",
        unit="epoch",
        disable=not show_progress,
    )

    for _ in epoch_iter:
        model.train()
        optimizer.zero_grad()

        weights = model(x_returns, x_priors, x_masks)
        replicated_return = (weights * y_returns * x_masks).sum(dim=-1)
        tracking_loss = torch.mean((replicated_return - y_index) ** 2)
        prior_active_mask = ((x_priors > 0) & (x_masks > 0)).float()
        log_ratio = torch.log(weights.clamp_min(1e-8)) - torch.log(
            x_priors.clamp_min(1e-8)
        )
        prior_loss = (log_ratio.square() * prior_active_mask).sum() / (
            prior_active_mask.sum().clamp_min(1.0)
        )
        smooth_mask = (x_masks[:, 1:, :] * x_masks[:, :-1, :]).float()
        weight_changes = weights[:, 1:, :] - weights[:, :-1, :]
        smoothness_loss = (weight_changes.square() * smooth_mask).sum() / (
            smooth_mask.sum().clamp_min(1.0)
        )

        loss = tracking_loss + l2_prior * prior_loss + l2_smoothness * smoothness_loss
        loss.backward()
        optimizer.step()

        if show_progress:
            epoch_iter.set_postfix(
                tracking_loss=f"{tracking_loss.item():.2e}",
                prior_loss=f"{prior_loss.item():.2e}",
                smoothness_loss=f"{smoothness_loss.item():.2e}",
            )

    model.eval()

    with torch.no_grad():
        weights = model(x_returns, x_priors, x_masks)

    if sequence_mode == "full":
        output_weights = weights[0].numpy()
    else:
        output_weights = weights[:, -1, :].numpy()

    weights_df = pd.DataFrame(
        output_weights,
        index=pd.Index(dates, name="date"),
        columns=columns,
    )

    return model, weights_df


def replicate_with_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    base_value: float = 100.0,
    base_index: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate returns by applying inferred prior-close weights to returns."""
    aligned_returns = returns.reindex(columns=weights.columns).fillna(0.0)
    weight_index = aligned_returns.index.union(weights.index).sort_values()
    aligned_weights = weights.reindex(index=weight_index, columns=weights.columns)
    contribution_weights = aligned_weights.shift(1).reindex(aligned_returns.index)
    valid_rows = contribution_weights.notna().any(axis=1)
    contribution_weights = contribution_weights.fillna(0.0)
    contributions = (aligned_returns * contribution_weights).loc[valid_rows]
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

    replicated_index = (1 + replicated_return.fillna(0)).cumprod() * start_value

    replicated = pd.DataFrame(
        {
            "replicated_return": replicated_return,
            "replicated_index": replicated_index,
        },
        index=replicated_return.index,
    )

    return replicated, contributions
