EXO-INSIGHT — M6 LOCAL QWEN INTEGRATION
======================================

WHAT THIS PACKAGE DOES
- Reproduces the M5 Logistic Regression pipeline with the same five features,
  preprocessing, 80/20 stratified split, random_state=42 and threshold=0.50.
- Saves the fitted model, metadata, held-out metrics and test predictions.
- Uses local Ollama Qwen3.5-4B Q4_K_M to explain a prediction.
- Keeps numeric ML outputs separate from Qwen's natural-language explanation.

SCIENTIFIC LIMIT
The target ml_target_conservative_hz is a protocol-derived M4 screening
pseudo-label. The score is a score for matching that label. It is NOT the
probability that a planet is habitable and is NOT evidence of life.

EXPECTED PROJECT PATH
S:\Aivancity\PGE 3\Coding for AI\EXO INSIGHTS

EXPECTED INPUT FILES
DATA\processed\exo_insight_preprocessed.csv
DATA\processed\milestone4\m4_ml_training_labels_review.csv

INSTALLATION
1. Extract this package into:
   S:\Aivancity\PGE 3\Coding for AI\EXO INSIGHTS\M6

2. In PowerShell:
   cd "S:\Aivancity\PGE 3\Coding for AI\EXO INSIGHTS\M6"

3. Reproduce/train and save the M5 pipeline:
   python train_m6_model.py

4. Run automated tests:
   python -m pytest -q

5. Explain a planet (replace with an exact name present in your CSV):
   python exo_insight_m6.py "Kepler-22 b"

OUTPUTS
M6\artifacts\m5_reproduced_pipeline.joblib
M6\artifacts\m6_model_metadata.json
M6\artifacts\m6_test_predictions.csv

REQUIREMENTS
Python 3.10+, pandas, numpy, scikit-learn, joblib, pytest; Ollama running
locally with qwen3.5:4b-q4_K_M available.

NOTES
- This package does not modify the M5 notebook.
- It deliberately uses only the five M5 model features.
- If your project folder is elsewhere, edit PROJECT_ROOT in train_m6_model.py.
- Training must run locally against your CSVs; no model is pre-trained in this ZIP.
