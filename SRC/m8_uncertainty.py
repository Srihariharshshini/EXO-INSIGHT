import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "artifacts" / "m6_test_predictions.csv"
OUTPUT = ROOT / "artifacts" / "m8_uncertainty_results.json"

N_BOOTSTRAP = 2000
RANDOM_SEED = 42

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing predictions: {INPUT}")

    df = pd.read_csv(INPUT)
    required = ["actual_screening_label", "predicted_screening_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    y_true = df["actual_screening_label"].astype(int).to_numpy()
    y_pred = df["predicted_screening_label"].astype(int).to_numpy()

    if len(y_true) == 0 or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Expected non-empty test data containing both classes.")

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)

    accuracy_samples = []
    balanced_samples = []

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        true_sample = y_true[indices]
        pred_sample = y_pred[indices]

        accuracy_samples.append(accuracy_score(true_sample, pred_sample))

        # Balanced accuracy requires both classes in the resampled sample.
        if len(np.unique(true_sample)) == 2:
            balanced_samples.append(
                balanced_accuracy_score(true_sample, pred_sample)
            )

    def interval(values):
        return {
            "mean": float(np.mean(values)),
            "ci_95_lower": float(np.percentile(values, 2.5)),
            "ci_95_upper": float(np.percentile(values, 97.5)),
            "bootstrap_standard_deviation": float(np.std(values, ddof=1))
        }

    results = {
        "project": "EXO-INSIGHT",
        "block": 8,
        "method": "Non-parametric bootstrap over held-out test rows",
        "bootstrap_repetitions": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "test_rows": n,
        "positive_reference_labels": int((y_true == 1).sum()),
        "negative_reference_labels": int((y_true == 0).sum()),
        "observed_accuracy": float(accuracy_score(y_true, y_pred)),
        "observed_balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "accuracy_bootstrap": interval(accuracy_samples),
        "balanced_accuracy_bootstrap": interval(balanced_samples),
        "balanced_accuracy_valid_resamples": len(balanced_samples),
        "limitation": (
            "These intervals describe sampling uncertainty on this held-out "
            "test sample. The reference labels are M4 protocol-derived "
            "pseudo-labels, not independently confirmed planetary "
            "habitability or evidence of life. The small number of positive "
            "test cases limits precision, especially for balanced accuracy."
        )
    }

    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== BLOCK 8: UNCERTAINTY RESULTS ===")
    print(json.dumps(results, indent=2))
    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()
