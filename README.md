# EXO-INSIGHT

A reproducible local AI and machine-learning project for analysing exoplanet screening data.

## Project overview

EXO-INSIGHT combines:
- NASA Exoplanet Archive data.
- Scientific Habitable Zone screening rules.
- A Logistic Regression classifier trained using M4 screening pseudo-labels.
- Local Qwen3.5-4B Q4_K_M through Ollama for evidence-based explanations.
- Automated tests and measured model comparisons.

## Scientific limitation

The classification target represents agreement with the M4 conservative Habitable Zone screening pseudo-label. It does not establish planetary habitability or evidence of life.

The model's label-match score is not the probability that a planet is habitable.

## Project structure

- DATA/raw: original dataset.
- DATA/processed: standardized data and M4 labels.
- SRC: training, prediction, Qwen integration and benchmarking scripts.
- TESTS: automated tests.
- artifacts: generated models, predictions, benchmark traces and metrics.
- EVIDENCE.md: evidence for the five selected building blocks.
- AI LOG: AI-use session notes and conversation exports.
- NOTEBOOKS: analysis notebooks.

## Requirements

- Python 3.10 or newer.
- Ollama installed and running.
- Qwen3.5-4B Q4_K_M.
- Python packages listed in requirements.txt.

## Installation

Open PowerShell in the project root.

Install Python dependencies:

    python -m pip install -r requirements.txt

Download the required model:

    ollama pull qwen3.5:4b-q4_K_M

Ensure Ollama is running and verify the model:

    ollama list

## Reproduce the machine-learning pipeline

Run:

    python SRC\train_m6_model.py

This loads the processed feature data and M4 labels, reproduces the M5 Logistic Regression pipeline, evaluates it on the held-out test set and saves the artifacts.

Generated files:
- artifacts/m5_reproduced_pipeline.joblib
- artifacts/m6_model_metadata.json
- artifacts/m6_test_predictions.csv

## Run automated tests

    python -m pytest TESTS -v

The tests cover feature schema, forbidden inputs, JSON-safe values, threshold decisions, invalid planet names, duplicate names and missing model artifacts.

## Run EXO-INSIGHT

With Ollama running and the model available:

    python SRC\exo_insight_m6.py "Kepler-22 b"

Replace the planet name with an exact name in DATA/processed/exo_insight_preprocessed.csv.

## Reproduce the M7 benchmark

    python SRC\run_m7_benchmark.py

This benchmark uses ten held-out cases, five from each reference class, with three Qwen-only runs per case. It compares the standalone Qwen predictions against the saved Logistic Regression predictions.

## Reproduce Block 8 uncertainty analysis

    python SRC\m8_uncertainty.py

This calculates bootstrap confidence intervals using 2,000 resamples.

## Data requirements

The following input files must be present:

    DATA\processed\exo_insight_preprocessed.csv
    DATA\processed\milestone4\m4_ml_training_labels_review.csv

The original NASA data and processed datasets must be included in the submitted repository or accompanied by reproducible acquisition and preprocessing instructions.

## Important implementation notes

- The classifier uses five features: pl_rade, pl_bmasse, disc_year, pl_bmassprov and discoverymethod.
- Random state: 42.
- Test split: 20%, stratified.
- Decision threshold: 0.50.
- Numeric missing values are median-imputed.
- Categorical missing values are most-frequent-imputed.
- Categorical variables are one-hot encoded.
- The classifier is Logistic Regression with class_weight="balanced".

## Model and data limitations

- M4 labels are protocol-derived pseudo-labels.
- The positive class is relatively small.
- Classification performance depends on the screening protocol and available features.
- The ten-task M7 comparison is exploratory and is not a population-level performance estimate.
- Qwen explanations must not replace the numerical classifier output.
- The local model is required for the final demonstration.

## Individual defence

The student must be able to explain the data preparation, scientific screening rules, model training, threshold, confusion matrix, Qwen integration, benchmark design and scientific limitations.
