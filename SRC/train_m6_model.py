from pathlib import Path
import json
import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = PROJECT_ROOT / "DATA" / "processed" / "exo_insight_preprocessed.csv"
LABELS_FILE = PROJECT_ROOT / "DATA" / "processed" / "milestone4" / "m4_ml_training_labels_review.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "m5_reproduced_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "m6_model_metadata.json"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "m6_test_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50
TARGET = "ml_target_conservative_hz"

NUMERIC_FEATURES = ["pl_rade", "pl_bmasse", "disc_year"]
CATEGORICAL_FEATURES = ["pl_bmassprov", "discoverymethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_FEATURES = [
    "pl_insol", "st_teff", "pl_eqt", "pl_orbper", "pl_orbsmax",
    "hz_runaway_greenhouse_flux", "hz_maximum_greenhouse_flux",
    "hz_screening_category", "insol_uncertainty_assessment"
]


def load_training_data():
    for file_path in (FEATURES_FILE, LABELS_FILE):
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")

    features = pd.read_csv(FEATURES_FILE)
    labels = pd.read_csv(LABELS_FILE)

    required_features = ["pl_name"] + FEATURES
    missing = [c for c in required_features if c not in features.columns]
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {missing}")

    if "pl_name" not in labels.columns or TARGET not in labels.columns:
        raise ValueError(
            f"Label dataset must contain pl_name and {TARGET}. "
            f"Available columns: {labels.columns.tolist()}"
        )

    if labels["pl_name"].duplicated().any():
        raise ValueError("Duplicate planet names in label dataset.")

    merged = labels[["pl_name", TARGET]].merge(
        features[required_features],
        on="pl_name",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        examples = merged.loc[
            merged["_merge"] != "both", "pl_name"
        ].head(10).tolist()
        raise ValueError(
            f"{unmatched} labels did not match feature data. Examples: {examples}"
        )

    merged = merged.drop(columns="_merge")
    merged[TARGET] = pd.to_numeric(merged[TARGET], errors="coerce")
    merged = merged.dropna(subset=[TARGET]).copy()
    merged[TARGET] = merged[TARGET].astype(int)

    if not set(merged[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary values 0 and 1.")

    return merged


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])


def main():
    print("EXO-INSIGHT M6 — M5 pipeline reproduction")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Feature file: {FEATURES_FILE}")
    print(f"Label file: {LABELS_FILE}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_training_data()

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Both target classes need at least two rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    model = build_pipeline()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    baseline_predictions = baseline.predict(X_test)

    metrics = {
        "rows": int(len(data)),
        "positive_labels": int((y == 1).sum()),
        "negative_labels": int((y == 0).sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "threshold": THRESHOLD,
        "baseline": {
            "accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, baseline_predictions)
            ),
            "precision": float(
                precision_score(y_test, baseline_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, baseline_predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, baseline_predictions, zero_division=0))
        },
        "logistic_regression": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "average_precision": float(
                average_precision_score(y_test, probabilities)
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist()
        },
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )
    }

    joblib.dump(model, MODEL_PATH)

    test_output = X_test.copy()
    test_output.insert(0, "pl_name", data.loc[X_test.index, "pl_name"])
    test_output["actual_screening_label"] = y_test
    test_output["predicted_screening_label"] = predictions
    test_output["predicted_label_match_score"] = probabilities
    test_output.to_csv(TEST_PREDICTIONS_PATH, index=False)

    metadata = {
        "project": "EXO-INSIGHT",
        "milestone": "M6",
        "model": "M5 LogisticRegression sklearn Pipeline",
        "target": TARGET,
        "target_interpretation": (
            "Agreement with M4 conservative HZ screening pseudo-label; "
            "not confirmed habitability or evidence of life."
        ),
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "forbidden_features": FORBIDDEN_FEATURES,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "scikit_learn_version": sklearn.__version__,
        "metrics": metrics
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nTRAINING COMPLETE")
    print(f"Rows: {len(data)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved predictions: {TEST_PREDICTIONS_PATH}")
    print("\nLogistic Regression metrics:")
    print(json.dumps(metrics["logistic_regression"], indent=2))

    print("\nScientific limitation:")
    print("The model predicts agreement with M4 screening labels,")
    print("not planetary habitability or the presence of life.")


if __name__ == "__main__":
    main()
