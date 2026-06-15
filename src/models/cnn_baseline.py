"""
CNN Trajectory Regressor baseline.
Input: task-space waypoint sequence (N, 3) -> Output: joint trajectory (N, 3).
"""

import torch
import torch.nn as nn


class CNNTrajectoryRegressor(nn.Module):
    """
    1D CNN that maps task-space waypoints to joint-space trajectories.

    Parameters
    ----------
    input_steps : int
        Number of input time steps.
    input_dim : int
        Dimension of input features per step (3 for x, y, theta).
    output_steps : int
        Number of output time steps.
    output_dim : int
        Dimension of output per step (3 for q1, q2, q3).
    conv_channels : list of int
        Number of channels for each conv layer.
    kernel_sizes : list of int
        Kernel size for each conv layer.
    fc_dims : list of int
        Hidden dimensions for FC layers.
    activation : str
        Activation function name.
    dropout : float
        Dropout rate.
    """

    def __init__(self, input_steps=100, input_dim=3, output_steps=100, output_dim=3,
                 conv_channels=None, kernel_sizes=None, fc_dims=None,
                 activation='relu', dropout=0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        if fc_dims is None:
            fc_dims = [256, 128]

        self.output_steps = output_steps
        self.output_dim = output_dim
        act = _get_activation(activation)

        # Build conv layers: input shape (B, input_dim, input_steps)
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
        self.conv = nn.Sequential(*conv_layers)

        # Adaptive pooling to fixed size then flatten
        self.pool = nn.AdaptiveAvgPool1d(1)

        # FC layers
        fc_layers = []
        fc_in = conv_channels[-1]
        for dim in fc_dims:
            fc_layers.extend([nn.Linear(fc_in, dim), act, nn.Dropout(dropout)])
            fc_in = dim
        fc_layers.append(nn.Linear(fc_in, output_steps * output_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor of shape (B, N, input_dim)

        Returns
        -------
        Tensor of shape (B, output_steps, output_dim)
        """
        # (B, N, C) -> (B, C, N) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, conv_channels[-1])
        x = self.fc(x)  # (B, output_steps * output_dim)
        return x.view(-1, self.output_steps, self.output_dim)


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