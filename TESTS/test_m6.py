import pandas as pd
from exo_insight_m6 import safe_value
from train_m6_model import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FORBIDDEN_FEATURES

def test_feature_schema_matches_m5():
    assert FEATURES == ["pl_rade", "pl_bmasse", "disc_year", "pl_bmassprov", "discoverymethod"]

def test_numeric_and_categorical_features_are_disjoint():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURES)

def test_forbidden_features_are_not_model_inputs():
    assert set(FEATURES).isdisjoint(FORBIDDEN_FEATURES)

def test_nan_becomes_json_safe_none():
    assert safe_value(float("nan")) is None

def test_numeric_values_remain_numeric():
    assert safe_value(4.2) == 4.2

def test_numpy_scalar_becomes_python_scalar():
    assert safe_value(pd.Series([3]).iloc[0]) == 3

# Additional M6 prediction and validation tests

import pytest
import exo_insight_m6 as m6
from pathlib import Path


class FakeModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, X):
        return __import__("numpy").array([[1 - self.score, self.score]])


def prepare_prediction_test(tmp_path, monkeypatch, score=0.50):
    features = pd.DataFrame([{
        "pl_name": "Test Planet b",
        "pl_rade": 1.2,
        "pl_bmasse": 2.5,
        "disc_year": 2020,
        "pl_bmassprov": "Mass",
        "discoverymethod": "Transit"
    }])

    feature_file = tmp_path / "features.csv"
    features.to_csv(feature_file, index=False)

    model_file = tmp_path / "model.joblib"
    model_file.touch()

    monkeypatch.setattr(m6, "FEATURES_FILE", feature_file)
    monkeypatch.setattr(m6, "MODEL_PATH", model_file)
    monkeypatch.setattr(m6.joblib, "load", lambda path: FakeModel(score))
    monkeypatch.setattr(
        m6, "generate_explanation",
        lambda evidence: "Test explanation"
    )

    return features


def test_prediction_at_threshold_is_positive(tmp_path, monkeypatch):
    prepare_prediction_test(tmp_path, monkeypatch, score=0.50)

    evidence, explanation = m6.explain_planet("Test Planet b")

    assert evidence["ml_prediction"]["predicted_screening_label"] == 1
    assert evidence["ml_prediction"]["label_match_score"] == 0.5


def test_prediction_below_threshold_is_negative(tmp_path, monkeypatch):
    prepare_prediction_test(tmp_path, monkeypatch, score=0.49)

    evidence, explanation = m6.explain_planet("Test Planet b")

    assert evidence["ml_prediction"]["predicted_screening_label"] == 0


def test_invalid_planet_name_is_rejected(tmp_path, monkeypatch):
    prepare_prediction_test(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="Planet not found"):
        m6.explain_planet("Unknown Planet")


def test_duplicate_planet_name_is_rejected(tmp_path, monkeypatch):
    features = prepare_prediction_test(tmp_path, monkeypatch)

    duplicated = pd.concat([features, features], ignore_index=True)
    duplicated.to_csv(m6.FEATURES_FILE, index=False)

    with pytest.raises(ValueError, match="not unique"):
        m6.explain_planet("Test Planet b")


def test_missing_model_artifact_is_rejected(tmp_path, monkeypatch):
    prepare_prediction_test(tmp_path, monkeypatch)

    monkeypatch.setattr(
        m6, "MODEL_PATH",
        tmp_path / "missing_model.joblib"
    )

    with pytest.raises(FileNotFoundError, match="Model not found"):
        m6.explain_planet("Test Planet b")


