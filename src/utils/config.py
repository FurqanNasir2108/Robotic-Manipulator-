"""
Configuration loading utilities.
"""

import yaml


def load_config(path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)