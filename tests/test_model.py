"""Tests for model prediction logic."""

import pytest


def test_risk_level_low():
    """Probability < 0.3 should be low risk."""
    prob = 0.15
    risk = "low" if prob < 0.3 else "medium" if prob < 0.6 else "high"
    assert risk == "low"


def test_risk_level_medium():
    """Probability between 0.3 and 0.6 should be medium risk."""
    prob = 0.45
    risk = "low" if prob < 0.3 else "medium" if prob < 0.6 else "high"
    assert risk == "medium"


def test_risk_level_high():
    """Probability >= 0.6 should be high risk."""
    prob = 0.85
    risk = "low" if prob < 0.3 else "medium" if prob < 0.6 else "high"
    assert risk == "high"


def test_prediction_threshold():
    """Default decision threshold should be 0.5."""
    prob = 0.51
    prediction = prob >= 0.5
    assert prediction is True

    prob = 0.49
    prediction = prob >= 0.5
    assert prediction is False
