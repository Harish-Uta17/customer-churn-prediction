"""
Basic tests for the Customer Churn Prediction pipeline.
Run with:  pytest
"""

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data.cleaner import clean_data

PROJECT_ROOT = Path(__file__).parent.parent


def test_config_loads():
    """The config file should load and return known keys."""
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    # test_size should exist and be a sensible fraction
    test_size = config.get("data.test_size", 0.2)
    assert 0 < test_size < 1


def test_raw_data_exists():
    """The raw dataset should be present in the repo."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "churn.csv"
    assert raw_path.is_file(), "Raw churn.csv is missing"


def test_clean_data_returns_dataframe():
    """clean_data should return a non-empty DataFrame with no fully-null rows."""
    raw_path = PROJECT_ROOT / "data" / "raw" / "churn.csv"
    df = pd.read_csv(raw_path)
    cleaned = clean_data(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) > 0
    assert cleaned.shape[1] > 0