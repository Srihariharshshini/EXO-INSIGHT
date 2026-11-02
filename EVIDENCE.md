# EXO-INSIGHT — EVIDENCE

Project: Coding for AI & Data Science, PGE3, 2026–2027

Selected building blocks: 4, 6, 7, 8 and 9.

---

## Block 4 — Decision model

### Implementation

A Logistic Regression classifier produces a score interpreted as the estimated probability of matching the M4 conservative HZ screening pseudo-label.

Decision threshold: 0.50.

Features:
- pl_rade
- pl_bmasse
- disc_year
- pl_bmassprov
- discoverymethod

### Evidence

Files:
- SRC/train_m6_model.py
- artifacts/m5_reproduced_pipeline.joblib
- artifacts/m6_model_metadata.json
- artifacts/m6_test_predictions.csv

Reproduction:

    python SRC\train_m6_model.py

Held-out test results:
- Test cases: 237
- Positive reference labels: 18
- Negative reference labels: 219
- True negatives: 149
- False positives: 70
- False negatives: 1
- True positives: 17

The threshold was set to 0.50 as an initial transparent decision threshold.

Limitation: reference labels are M4 protocol-derived pseudo-labels, not confirmed habitability labels.

---

## Block 6 — Messy data and documented decisions

### Data source

NASA Exoplanet Archive, PSCompPars dataset.

Original dataset:
- 6,451 rows
- 71 columns
- 6,451 unique planet names
- Discovery years 1992–2026

### Cleaning and preparation

The standardized analysis dataset contains 6,451 rows and 15 selected research columns.

- No rows were removed.
- No missing values were imputed in the standardized analysis dataset.
- Missing values were retained for downstream handling.
- Duplicate planet-name rows were not identified.
- The st_lum research column was unavailable.
- 46 zero-insolation records were retained but excluded from HZ classification.
- These records were associated with imaging discoveries and pl_insollim=0.

### Scientific rule obtained from a document

The Habitable Zone screening uses the method and coefficients from Kopparapu et al. (2014), including conservative runaway-greenhouse and maximum-greenhouse boundaries, and optimistic Recent Venus and Early Mars boundaries.

The host-star effective-temperature applicability range used was 2600–7200 K.

This scientific rule is not contained in the raw CSV as a ready-made classification. It was obtained from the scientific reference and applied during analysis.

### M4 candidate review

- Candidate review queue: 254.
- Tier 1: 62.
- Tier 2: 10.
- Tier 3: 17.
- Tier 4: 72.
- Tier 5: 93.

The M4 machine-learning label dataset contains 1,184 rows:
- Positive: 89.
- Negative: 1,095.

Evidence files include the original dataset, standardized data, M4 analysis outputs, candidate review data and the cleaning records in DATA and NOTEBOOKS.

Reproduction depends on the original data and preprocessing workflow being included in the submitted repository.

---

## Block 7 — Model learned from data

### Model

Logistic Regression with:
- Median imputation and standard scaling for numeric features.
- Most-frequent imputation and one-hot encoding for categorical features.
- Balanced class weights.
- Random state 42.
- Stratified 80/20 train-test split.

### Baseline comparison

Majority-class baseline:
- Accuracy: 0.9241.
- Balanced accuracy: 0.5000.
- Positive-class recall: 0.0000.

Logistic Regression:
- Accuracy: 0.7004.
- Balanced accuracy: 0.8124.
- Precision: 0.1954.
- Recall: 0.9444.
- F1: 0.3238.
- Average precision: 0.1752.

Confusion matrix:

    [[149, 70],
     [  1, 17]]

The classifier improves balanced accuracy and positive-class recall compared with the majority baseline, while generating many false positives.

Evidence:
- SRC/train_m6_model.py
- artifacts/m6_model_metadata.json
- artifacts/m6_test_predictions.csv

Reproduction:

    python SRC\train_m6_model.py

Limitation: these metrics measure agreement with protocol-derived pseudo-labels, not confirmed planetary habitability.

---

## Block 8 — Numbers with uncertainty

### Method

Non-parametric bootstrap over the 237 held-out test rows.

- Bootstrap repetitions: 2,000.
- Random seed: 42.
- Positive reference labels: 18.
- Negative reference labels: 219.

### Results

Accuracy:
- Observed: 0.7004.
- 95% bootstrap confidence interval: 0.6414–0.7595.
- Bootstrap standard deviation: 0.0298.

Balanced accuracy:
- Observed: 0.8124.
- 95% bootstrap confidence interval: 0.7422–0.8641.
- Bootstrap standard deviation: 0.0318.

Evidence:
- SRC/m8_uncertainty.py
- artifacts/m8_uncertainty_results.json

Reproduction:

    python SRC\m8_uncertainty.py

Limitation: the intervals describe sampling uncertainty against M4 pseudo-labels. They do not measure uncertainty about actual habitability or life. The small positive test class limits precision.

---

## Block 9 — Measured comparison

### Benchmark design

Compared:
1. EXO-INSIGHT Logistic Regression classifier.
2. Standalone local Qwen3.5-4B Q4_K_M predicting the same M4 reference labels from the same five input features.

Design:
- 10 held-out tasks.
- 5 positive and 5 negative reference cases.
- 3 runs per task.
- 30 Qwen calls.
- All 30 Qwen responses were valid.
- Temperature: 0.

### Results

EXO-INSIGHT:
- Run accuracies: 0.90, 0.90, 0.90.
- Mean accuracy: 0.90.
- Standard deviation: 0.00.

Qwen alone:
- Run accuracies: 0.50, 0.50, 0.50.
- Mean accuracy: 0.50.
- Standard deviation: 0.00.

### Failure trace 1 — Qwen alone

Planet: HD 83443 c.

- Reference label: 0.
- Qwen prediction: 1.
- Raw response: 1.
- Radius: 13.6.
- Mass: 476.7426105.
- Discovery year: 2022.
- Mass provenance: Mass.
- Discovery method: Radial Velocity.

Qwen returned label 1 for this negative reference case. The trace records the wrong prediction but does not reveal the model's internal reasoning.

### Failure trace 2 — EXO-INSIGHT

Planet: HD 103720 b.

- Reference label: 0.
- EXO-INSIGHT prediction: 1.
- Radius: 14.1.
- Mass: 197.046.
- Discovery year: 2014.
- Mass provenance: Msini.
- Discovery method: Radial Velocity.

This is a false positive against the M4 pseudo-label.

Evidence:
- SRC/run_m7_benchmark.py
- artifacts/m7_benchmark_traces.csv
- artifacts/m7_benchmark_summary.json

Reproduction:

    python SRC\run_m7_benchmark.py

Limitation: ten tasks form a small exploratory benchmark. Results should not be generalized to the full exoplanet population. The repeated zero standard deviation reflects deterministic repeated runs at temperature zero.

---

## R1 — Reproducibility

Main instructions are provided in README.md.

Required inputs:
- DATA/processed/exo_insight_preprocessed.csv
- DATA/processed/milestone4/m4_ml_training_labels_review.csv

Required local model:
- qwen3.5:4b-q4_K_M

Reproduction commands:

    python -m pip install -r requirements.txt
    ollama pull qwen3.5:4b-q4_K_M
    python SRC\train_m6_model.py
    python -m pytest TESTS -v
    python SRC\exo_insight_m6.py "Kepler-22 b"

A fresh-clone reproduction must be verified before claiming R1 fully passed.

---

## R2 — Automated tests

Test file:
- TESTS/test_m6.py
- TESTS/conftest.py

Verified local result:
- 11 tests passed.

Command:

    python -m pytest TESTS -v

The tests cover schema restrictions, JSON-safe values, threshold boundaries, invalid names, duplicate names and missing model artifacts.

Fresh-clone test execution remains to be verified.

---

## R3 — AI-use log

Session notes are stored in the AI LOG folder.

Conversation exports must also be included in that folder before final submission. Do not treat session notes alone as complete R3 evidence.

---

## R4 — Individual defence

The student must personally run the system and explain code selected by the teacher.

Defence preparation should cover:
- Data cleaning.
- Scientific HZ screening.
- Logistic Regression and preprocessing.
- Threshold decisions.
- Confusion matrix.
- Qwen integration.
- Benchmark methodology.
- Scientific limitations.
