"""
CNN + LSTM Trajectory Regressor.
CNN feature extractor -> LSTM sequence decoder -> joint trajectory.
"""

import torch
import torch.nn as nn


class CNNLSTMRegressor(nn.Module):
    """
    CNN encoder followed by LSTM decoder for trajectory prediction.

    Parameters
    ----------
    input_steps : int
    input_dim : int
    output_steps : int
    output_dim : int
    conv_channels : list of int
    kernel_sizes : list of int
    lstm_hidden : int
    lstm_layers : int
    lstm_dropout : float
    fc_dims : list of int
    activation : str
    dropout : float
    """

    def __init__(self, input_steps=100, input_dim=3, output_steps=100, output_dim=3,
                 conv_channels=None, kernel_sizes=None, lstm_hidden=128,
                 lstm_layers=2, lstm_dropout=0.1, fc_dims=None,
                 activation='relu', dropout=0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]
        if fc_dims is None:
            fc_dims = [64]

        self.output_steps = output_steps
        self.output_dim = output_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        act = _get_activation(activation)

        # CNN encoder
        conv_layers = []
        in_ch = input_dim
        for ch, ks in zip(conv_channels, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                act,
                nn.Dropout(dropout),
            ])
            in_ch = ch
        self.encoder = nn.Sequential(*conv_layers)

        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # Output projection
        fc_layers = []
        fc_in = lstm_hidden
        for dim in fc_dims:
            fc_layers.extend([nn.Linear(fc_in, dim), act, nn.Dropout(dropout)])
            fc_in = dim
        fc_layers.append(nn.Linear(fc_in, output_dim))
        self.output_proj = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (B, N, input_dim)

        Returns
        -------
        Tensor of shape (B, output_steps, output_dim)
        """
        # CNN encode: (B, N, C) -> (B, C, N) -> (B, C_out, N)
        enc = self.encoder(x.permute(0, 2, 1))
        # (B, C_out, N) -> (B, N, C_out)
        enc = enc.permute(0, 2, 1)

        # LSTM decode
        lstm_out, _ = self.lstm(enc)  # (B, N, lstm_hidden)

        # Project each time step to output_dim
        out = self.output_proj(lstm_out)  # (B, N, output_dim)
        return out


def _get_activation(name):
    """Return activation module by name."""
    activations = {
        'relu': nn.ReLU(inplace=True),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'tanh': nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]