"""
Conditional Variational Autoencoder (cVAE) for trajectory generation.

Supports:
- Condition encoder: 1D-CNN on waypoint sequences
- VAE encoder: BiGRU or 1D-CNN → (mu, log_var)
- VAE decoder: GRU or 1D-CNN → joint trajectory
"""

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """Encode task-space waypoints into a fixed-size condition embedding via 1D-CNN."""

    def __init__(self, input_dim=3, conv_channels=None, kernel_sizes=None,
                 condition_dim=64, dropout=0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        layers = []
        in_ch = input_dim
        for ch, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(conv_channels[-1], condition_dim)

    def forward(self, waypoints):
        """
        Parameters
        ----------
        waypoints : (B, T, 3) task-space trajectory

        Returns
        -------
        cond : (B, condition_dim)
        """
        x = waypoints.permute(0, 2, 1)  # (B, 3, T)
        x = self.conv(x)                # (B, C, T)
        x = self.pool(x).squeeze(-1)    # (B, C)
        return self.proj(x)             # (B, condition_dim)


class BiGRUEncoder(nn.Module):
    """VAE encoder: BiGRU over concatenated condition + trajectory → (mu, log_var)."""

    def __init__(self, input_dim=3, condition_dim=64, hidden=128, layers=2,
                 dropout=0.1, latent_dim=16):
        super().__init__()
        self.cond_proj = nn.Linear(condition_dim, input_dim)
        self.gru = nn.GRU(
            input_size=input_dim * 2,  # trajectory + projected condition
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.mu_proj = nn.Linear(hidden * 2, latent_dim)
        self.logvar_proj = nn.Linear(hidden * 2, latent_dim)

    def forward(self, trajectory, cond):
        """
        Parameters
        ----------
        trajectory : (B, T, 3) ground-truth joint trajectory
        cond : (B, condition_dim)

        Returns
        -------
        mu : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        B, T, _ = trajectory.shape
        cond_expanded = self.cond_proj(cond).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([trajectory, cond_expanded], dim=-1)  # (B, T, 6)
        _, h = self.gru(x)  # h: (2*layers, B, hidden)
        # Take the last layer's forward and backward hidden states
        h_fwd = h[-2]   # (B, hidden)
        h_bwd = h[-1]   # (B, hidden)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*hidden)
        return self.mu_proj(h_cat), self.logvar_proj(h_cat)


class CNNEncoder(nn.Module):
    """VAE encoder: 1D-CNN over concatenated condition + trajectory → (mu, log_var)."""

    def __init__(self, input_dim=3, condition_dim=64, conv_channels=None,
                 kernel_sizes=None, dropout=0.1, latent_dim=16):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        self.cond_proj = nn.Linear(condition_dim, input_dim)

        layers = []
        in_ch = input_dim * 2
        for ch, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mu_proj = nn.Linear(conv_channels[-1], latent_dim)
        self.logvar_proj = nn.Linear(conv_channels[-1], latent_dim)

    def forward(self, trajectory, cond):
        B, T, _ = trajectory.shape
        cond_expanded = self.cond_proj(cond).unsqueeze(1).expand(B, T, -1)
        x = torch.cat([trajectory, cond_expanded], dim=-1)  # (B, T, 6)
        x = x.permute(0, 2, 1)   # (B, 6, T)
        x = self.conv(x)         # (B, C, T)
        x = self.pool(x).squeeze(-1)  # (B, C)
        return self.mu_proj(x), self.logvar_proj(x)


class GRUDecoder(nn.Module):
    """VAE decoder: GRU conditioned on z + condition → joint trajectory."""

    def __init__(self, latent_dim=16, condition_dim=64, output_dim=3,
                 output_steps=100, hidden=128, layers=2, dropout=0.1,
                 fc_dims=None):
        super().__init__()
        if fc_dims is None:
            fc_dims = [64]

        self.output_steps = output_steps
        self.input_proj = nn.Linear(latent_dim + condition_dim, hidden)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )

        out_layers = []
        fc_in = hidden
        for dim in fc_dims:
            out_layers.extend([nn.Linear(fc_in, dim), nn.ReLU(inplace=True),
                               nn.Dropout(dropout)])
            fc_in = dim
        out_layers.append(nn.Linear(fc_in, output_dim))
        self.output_proj = nn.Sequential(*out_layers)

    def forward(self, z, cond):
        """
        Parameters
        ----------
        z : (B, latent_dim)
        cond : (B, condition_dim)

        Returns
        -------
        trajectory : (B, T, output_dim)
        """
        B = z.shape[0]
        zc = torch.cat([z, cond], dim=-1)           # (B, latent_dim + condition_dim)
        inp = self.input_proj(zc)                     # (B, hidden)
        inp = inp.unsqueeze(1).expand(B, self.output_steps, -1)  # (B, T, hidden)
        out, _ = self.gru(inp)                        # (B, T, hidden)
        return self.output_proj(out)                  # (B, T, output_dim)


class CNNDecoder(nn.Module):
    """VAE decoder: 1D-CNN conditioned on z + condition → joint trajectory."""

    def __init__(self, latent_dim=16, condition_dim=64, output_dim=3,
                 output_steps=100, conv_channels=None, kernel_sizes=None,
                 dropout=0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [128, 64, 32]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]

        self.output_steps = output_steps
        self.input_proj = nn.Linear(latent_dim + condition_dim, conv_channels[0] * output_steps)
        self.init_channels = conv_channels[0]

        layers = []
        in_ch = conv_channels[0]
        for ch, ks in zip(conv_channels[1:], kernel_sizes[1:]):
            layers.extend([
                nn.Conv1d(in_ch, ch, ks, padding=ks // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_ch = ch
        layers.append(nn.Conv1d(in_ch, output_dim, 1))
        self.conv = nn.Sequential(*layers)

    def forward(self, z, cond):
        B = z.shape[0]
        zc = torch.cat([z, cond], dim=-1)
        x = self.input_proj(zc)                          # (B, C0*T)
        x = x.view(B, self.init_channels, self.output_steps)  # (B, C0, T)
        x = self.conv(x)                                  # (B, output_dim, T)
        return x.permute(0, 2, 1)                         # (B, T, output_dim)


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for trajectory generation.

    Encodes waypoints → condition embedding, then encodes
    (condition + ground-truth trajectory) → latent z, then decodes
    (z + condition) → predicted joint trajectory.
    """

    def __init__(self, latent_dim=16, encoder_type='bigru', decoder_type='gru',
                 condition_dim=64, input_dim=3, output_dim=3,
                 input_steps=100, output_steps=100,
                 cond_conv_channels=None, cond_kernel_sizes=None,
                 enc_hidden=128, enc_layers=2, enc_dropout=0.1,
                 dec_hidden=128, dec_layers=2, dec_dropout=0.1,
                 dec_fc_dims=None, dec_conv_channels=None, dec_kernel_sizes=None):
        super().__init__()
        self.latent_dim = latent_dim

        # Condition encoder
        self.cond_encoder = ConditionEncoder(
            input_dim=input_dim, conv_channels=cond_conv_channels,
            kernel_sizes=cond_kernel_sizes, condition_dim=condition_dim,
            dropout=enc_dropout,
        )

        # VAE encoder
        if encoder_type == 'bigru':
            self.encoder = BiGRUEncoder(
                input_dim=output_dim, condition_dim=condition_dim,
                hidden=enc_hidden, layers=enc_layers,
                dropout=enc_dropout, latent_dim=latent_dim,
            )
        elif encoder_type == 'cnn':
            self.encoder = CNNEncoder(
                input_dim=output_dim, condition_dim=condition_dim,
                dropout=enc_dropout, latent_dim=latent_dim,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # VAE decoder
        if decoder_type == 'gru':
            self.decoder = GRUDecoder(
                latent_dim=latent_dim, condition_dim=condition_dim,
                output_dim=output_dim, output_steps=output_steps,
                hidden=dec_hidden, layers=dec_layers, dropout=dec_dropout,
                fc_dims=dec_fc_dims,
            )
        elif decoder_type == 'cnn':
            self.decoder = CNNDecoder(
                latent_dim=latent_dim, condition_dim=condition_dim,
                output_dim=output_dim, output_steps=output_steps,
                conv_channels=dec_conv_channels, kernel_sizes=dec_kernel_sizes,
                dropout=dec_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def encode(self, trajectory, cond):
        """Encode trajectory + condition → (mu, log_var)."""
        return self.encoder(trajectory, cond)

    def reparameterize(self, mu, log_var):
        """Sample z ~ N(mu, sigma) using reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        """Decode z + condition → trajectory."""
        return self.decoder(z, cond)

    def forward(self, waypoints, trajectory):
        """
        Full forward pass (training).

        Parameters
        ----------
        waypoints : (B, T, 3) task-space waypoints
        trajectory : (B, T, 3) ground-truth joint trajectory

        Returns
        -------
        recon : (B, T, 3) reconstructed trajectory
        mu : (B, latent_dim)
        log_var : (B, latent_dim)
        """
        cond = self.cond_encoder(waypoints)
        mu, log_var = self.encode(trajectory, cond)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, cond)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, waypoints, num_samples=1):
        """
        Generate trajectories from prior z ~ N(0, I).

        Parameters
        ----------
        waypoints : (B, T, 3) or (1, T, 3)
        num_samples : int — number of samples per condition

        Returns
        -------
        trajectories : (B * num_samples, T, 3)
        """
        self.eval()
        cond = self.cond_encoder(waypoints)  # (B, condition_dim)
        B = cond.shape[0]
        if num_samples > 1:
            cond = cond.unsqueeze(1).expand(B, num_samples, -1).reshape(B * num_samples, -1)
        z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device)
        return self.decode(z, cond)