"""
CNN + GRU Trajectory Regressor.
CNN feature extractor -> GRU sequence decoder -> joint trajectory.
"""

import torch
import torch.nn as nn


class CNNGRURegressor(nn.Module):
    """
    CNN encoder followed by GRU decoder for trajectory prediction.

    Parameters
    ----------
    input_steps : int
    input_dim : int
    output_steps : int
    output_dim : int
    conv_channels : list of int
    kernel_sizes : list of int
    gru_hidden : int
    gru_layers : int
    gru_dropout : float
    fc_dims : list of int
    activation : str
    dropout : float
    """

    def __init__(self, input_steps=100, input_dim=3, output_steps=100, output_dim=3,
                 conv_channels=None, kernel_sizes=None, gru_hidden=128,
                 gru_layers=2, gru_dropout=0.1, fc_dims=None,
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

        # GRU decoder
        self.gru = nn.GRU(
            input_size=conv_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0,
        )

        # Output projection
        fc_layers = []
        fc_in = gru_hidden
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
        enc = self.encoder(x.permute(0, 2, 1))
        enc = enc.permute(0, 2, 1)
        gru_out, _ = self.gru(enc)
        out = self.output_proj(gru_out)
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