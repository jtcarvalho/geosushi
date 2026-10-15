"""
YAML configuration helpers shared by all geosushi tools.

Example
-------
>>> from geosushi.common.config import load_config, cfg_get
>>> cfg = load_config("grid.yaml")
>>> dx = cfg_get(cfg, "grid.dx", 0.1)
"""

import os

import yaml


def load_config(config_file):
    """Read a YAML file and return it as a dict (raises if it does not exist)."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def cfg_get(config, key_path, default=None):
    """Get a nested value using dot notation, e.g. ``cfg_get(cfg, 'grid.dx')``."""
    if config is None:
        return default
    value = config
    for key in key_path.split("."):
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    return value if value is not None else default
