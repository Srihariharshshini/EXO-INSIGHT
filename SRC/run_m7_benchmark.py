import csv
import json
import re
import time
import random
import statistics
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "artifacts" / "m6_test_predictions.csv"
OUTPUT = ROOT / "artifacts"
TRACE_FILE = OUTPUT / "m7_benchmark_traces.csv"
SUMMARY_FILE = OUTPUT / "m7_benchmark_summary.json"

MODEL = "qwen3.5:4b-q4_K_M"
URL = "http://localhost:11434/api/generate"
RUNS = 3
TASKS_PER_CLASS = 5
RANDOM_SEED = 42
FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "disc_year",
    "pl_bmassprov",
    "discoverymethod",
]

def qwen_predict(features):
    prompt = f"""
You are completing a binary classification benchmark.

Predict the reference screening label using ONLY these supplied features:
{json.dumps(features, ensure_ascii=False)}

The label definitions are:
0 = non-agreement with the M4 conservative HZ screening pseudo-label
1 = agreement with the M4 conservative HZ screening pseudo-label

Return exactly one character: 0 or 1.
Do not explain your answer. Do not return any other text.
"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 20
        }
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    raw = result.get("response", "").strip()
    match = re.fullmatch(r"\s*([01])\s*", raw)
    prediction = int(match.group(1)) if match else None
    return prediction, raw, elapsed

def safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing M6 predictions: {INPUT}")

    df = pd.read_csv(INPUT)
    required = FEATURES + [
        "pl_name",
        "actual_screening_label",
        "predicted_screening_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["actual_screening_label"]).copy()
    df["actual_screening_label"] = df["actual_screening_label"].astype(int)
    df["predicted_screening_label"] = df["predicted_screening_label"].astype(int)

    if not set(df["actual_screening_label"].unique()).issubset({0, 1}):
        raise ValueError("Reference labels must be binary 0/1.")

    selected_parts = []
    for label in [0, 1]:
        group = df[df["actual_screening_label"] == label]
        if len(group) < TASKS_PER_CLASS:
            raise ValueError(f"Not enough cases for reference label {label}.")
        selected_parts.append(
            group.sample(n=TASKS_PER_CLASS, random_state=RANDOM_SEED)
        )

    tasks = pd.concat(selected_parts).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Selected {len(tasks)} benchmark tasks.")
    print("Reference labels:", tasks["actual_screening_label"].value_counts().to_dict())
    print(f"Running {RUNS} Qwen repetitions per task: {len(tasks) * RUNS} calls.")
    print("This may take several minutes. Keep Ollama running.")

    traces = []
    qwen_run_accuracies = []
    system_run_accuracies = []
    qwen_latencies = []

    for run_number in range(1, RUNS + 1):
        qwen_correct = 0
        system_correct = 0
        qwen_valid = 0

        print(f"\n=== RUN {run_number}/{RUNS} ===")

        for _, row in tasks.iterrows():
            task_id = str(row["pl_name"])
            actual = int(row["actual_screening_label"])
            system_prediction = int(row["predicted_screening_label"])
            features = {name: safe_value(row[name]) for name in FEATURES}

            system_correct += int(system_prediction == actual)

            try:
                qwen_prediction, raw, elapsed = qwen_predict(features)
                qwen_latencies.append(elapsed)
                if qwen_prediction is not None:
                    qwen_valid += 1
                    qwen_correct += int(qwen_prediction == actual)
                error = ""
            except Exception as exc:
                qwen_prediction = None
                raw = ""
                elapsed = None
                error = f"{type(exc).__name__}: {exc}"

            traces.append({
                "task_id": task_id,
                "run": run_number,
                "actual_label": actual,
                "exo_insight_prediction": system_prediction,
                "exo_insight_correct": int(system_prediction == actual),
                "qwen_prediction": qwen_prediction,
                "qwen_correct": (
                    int(qwen_prediction == actual)
                    if qwen_prediction is not None else 0
                ),
                "qwen_valid": int(qwen_prediction is not None),
                "qwen_latency_seconds": elapsed,
                "qwen_raw_response": raw,
                "error": error,
                "features_json": json.dumps(features, ensure_ascii=False),
            })

            status = (
                "VALID" if qwen_prediction is not None else "INVALID"
            )
            print(
                f"{task_id}: actual={actual}, "
                f"EXO={system_prediction}, "
                f"Qwen={qwen_prediction} ({status})"
            )

        system_run_accuracies.append(system_correct / len(tasks))
        qwen_run_accuracies.append(qwen_correct / len(tasks))

        print(
            f"Run {run_number}: "
            f"EXO-INSIGHT accuracy={system_correct / len(tasks):.3f}; "
            f"Qwen accuracy={qwen_correct / len(tasks):.3f}"
        )

    with TRACE_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=traces[0].keys())
        writer.writeheader()
        writer.writerows(traces)

    qwen_predictions = [t["qwen_prediction"] for t in traces]
    qwen_correct_flags = [t["qwen_correct"] for t in traces]
    system_predictions = [t["exo_insight_prediction"] for t in traces]
    actual_labels = [t["actual_label"] for t in traces]

    def spread(values):
        return statistics.stdev(values) if len(values) > 1 else 0.0

    summary = {
        "project": "EXO-INSIGHT",
        "milestone": "M7",
        "benchmark": "EXO-INSIGHT Logistic Regression vs standalone local Qwen",
        "model": MODEL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "task_count": len(tasks),
        "reference_label_counts": {
            str(k): int(v)
            for k, v in tasks["actual_screening_label"].value_counts().items()
        },
        "runs_per_task": RUNS,
        "total_qwen_calls": len(traces),
        "qwen_valid_responses": int(sum(t["qwen_valid"] for t in traces)),
        "qwen_invalid_responses": int(
            sum(not t["qwen_valid"] for t in traces)
        ),
        "exo_insight": {
            "run_accuracies": system_run_accuracies,
            "mean_accuracy": statistics.mean(system_run_accuracies),
            "accuracy_standard_deviation": spread(system_run_accuracies),
            "correct_predictions": int(sum(
                int(p == a) for p, a in zip(system_predictions, actual_labels)
            )),
            "total_predictions": len(system_predictions),
        },
        "qwen_alone": {
            "run_accuracies": qwen_run_accuracies,
            "mean_accuracy": statistics.mean(qwen_run_accuracies),
            "accuracy_standard_deviation": spread(qwen_run_accuracies),
            "correct_predictions": int(sum(qwen_correct_flags)),
            "total_predictions": len(traces),
            "valid_response_rate": sum(t["qwen_valid"] for t in traces) / len(traces),
            "mean_latency_seconds": (
                statistics.mean(qwen_latencies) if qwen_latencies else None
            ),
            "latency_standard_deviation_seconds": (
                spread(qwen_latencies) if len(qwen_latencies) > 1 else None
            ),
        },
        "failure_examples": [
            {
                "task_id": t["task_id"],
                "run": t["run"],
                "actual_label": t["actual_label"],
                "qwen_prediction": t["qwen_prediction"],
                "raw_response": t["qwen_raw_response"],
                "error": t["error"],
            }
            for t in traces
            if t["qwen_prediction"] != t["actual_label"]
        ][:5],
        "limitations": [
            "Only 10 held-out cases were selected; results are an exploratory benchmark, not a population-level estimate.",
            "The reference answers are M4 protocol-derived pseudo-labels, not confirmed habitability or evidence of life.",
            "The Qwen-only system receives the same five features but does not receive the trained classifier's prediction.",
            "The EXO-INSIGHT classifier is deterministic; repeated runs measure reproducibility, not independent retraining."
        ],
        "trace_file": str(TRACE_FILE.relative_to(ROOT)),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== M7 BENCHMARK COMPLETE ===")
    print("Trace:", TRACE_FILE)
    print("Summary:", SUMMARY_FILE)
    print("\nEXO-INSIGHT mean accuracy:", round(summary["exo_insight"]["mean_accuracy"], 4))
    print("EXO-INSIGHT accuracy spread (SD):", round(summary["exo_insight"]["accuracy_standard_deviation"], 4))
    print("Qwen-alone mean accuracy:", round(summary["qwen_alone"]["mean_accuracy"], 4))
    print("Qwen-alone accuracy spread (SD):", round(summary["qwen_alone"]["accuracy_standard_deviation"], 4))
    print("Qwen valid responses:", summary["qwen_valid_responses"], "/", summary["total_qwen_calls"])
    print("M7 raw traces and summary saved.")

if __name__ == "__main__":
    main()
