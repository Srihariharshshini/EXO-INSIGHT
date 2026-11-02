import argparse
import json
import math
import joblib
import pandas as pd
try:
    from .train_m6_model import (
        FEATURES_FILE, FEATURES, MODEL_PATH, THRESHOLD as INITIAL_THRESHOLD
    )
except ImportError:
    from train_m6_model import (
        FEATURES_FILE, FEATURES, MODEL_PATH, THRESHOLD as INITIAL_THRESHOLD
    )
try:
    from .m6_qwen import generate_explanation
except ImportError:
    from m6_qwen import generate_explanation

def safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def explain_planet(planet_name):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run: python train_m6_model.py")
    data = pd.read_csv(FEATURES_FILE)
    required = ["pl_name"] + FEATURES
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Feature CSV missing columns: {missing}")
    matches = data.loc[data["pl_name"].astype(str).str.casefold() == planet_name.casefold()]
    if matches.empty:
        raise ValueError(f"Planet not found by exact name: {planet_name}")
    if len(matches) != 1:
        raise ValueError(f"Planet name is not unique: {planet_name}")
    row = matches.iloc[0]
    X = pd.DataFrame([{f: row[f] for f in FEATURES}], columns=FEATURES)
    model = joblib.load(MODEL_PATH)
    score = float(model.predict_proba(X)[0, 1])
    prediction = int(score >= INITIAL_THRESHOLD)
    evidence = {
        "planet_name": str(row["pl_name"]),
        "ml_prediction": {
            "predicted_screening_label": prediction,
            "label_meaning": "1 = predicts agreement with M4 conservative HZ screening pseudo-label; 0 = predicts non-agreement",
            "label_match_score": round(score, 6), "threshold": INITIAL_THRESHOLD
        },
        "input_features": {f: safe_value(row[f]) for f in FEATURES},
        "input_feature_units": {
            "pl_rade": "Earth radii",
            "pl_bmasse": "Earth masses",
            "disc_year": "calendar year",
            "pl_bmassprov": "categorical value; no physical unit",
            "discoverymethod": "categorical value; no physical unit"
        },
        "scientific_limit": "Prediction against protocol-derived M4 pseudo-labels; does not establish habitability or life."
    }
    return evidence, generate_explanation(evidence)

def main():
    parser = argparse.ArgumentParser(description="EXO-INSIGHT M6 local Qwen explanation")
    parser.add_argument("planet", help='Exact planet name, e.g. "Kepler-22 b"')
    args = parser.parse_args()
    evidence, explanation = explain_planet(args.planet)
    print("\\n=== EXO-INSIGHT STRUCTURED RESULT ===")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    print("\\n=== LOCAL QWEN EXPLANATION ===")
    print(explanation)

if __name__ == "__main__":
    main()



