"""Edge deployment configuration loader."""

from __future__ import annotations

import yaml


def load_edge_config(config_path: str = "configs/edge_deployment.yaml") -> dict:
    """Load edge deployment configuration.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
