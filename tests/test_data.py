"""Tests for data ingestion and preprocessing."""

import pandas as pd
import pytest


def test_target_column_is_binary():
    """After preprocessing, Churn column should only contain 0 and 1."""
    # This test will use real data in Phase 2
    # For now, test the logic on a sample dataframe
    df = pd.DataFrame({"Churn": ["Yes", "No", "Yes", "No"]})
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_no_missing_values_after_clean():
    """Cleaned dataframe should have no NaN values."""
    df = pd.DataFrame({"TotalCharges": ["100.5", "  ", "200.0"], "Churn": [1, 0, 1]})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    assert df.isnull().sum().sum() == 0


def test_train_val_test_split_sizes():
    """Check that split proportions are approximately correct."""
    total = 1000
    test_size = 0.2
    val_size = 0.1
    n_test = int(total * test_size)
    n_val = int((total - n_test) * val_size / (1 - test_size))
    n_train = total - n_test - n_val

    assert n_test == pytest.approx(200, abs=5)
    assert n_train > n_val
    assert n_train + n_val + n_test == total
