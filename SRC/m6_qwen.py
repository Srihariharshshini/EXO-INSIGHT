import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3.5:4b-q4_K_M"

def generate_explanation(evidence, timeout=300):

    prompt = f"""You are the explanation component of EXO-INSIGHT.

Explain the supplied JSON evidence accurately.

STRICT RULES:
1. Use only facts explicitly supplied in the JSON.
2. Never expand or guess the meaning of abbreviations such as Msini.
3. Msini must remain exactly as supplied. Do not call it minimum stellar mass.
4. Call the numerical score a label-match score, never model confidence.
5. The score measures agreement with an M4 screening pseudo-label only.
6. Never claim that a planet is habitable or contains life.
7. Never invent measurements, properties, or scientific conclusions.
8. Never change or override the numerical prediction.
9. If information is missing, say it is unavailable.


IMPORTANT SCIENTIFIC FIELD DEFINITIONS:
- pl_bmasse means PLANETARY MASS, not host-star mass. Its unit is Earth masses.
- pl_rade means PLANETARY RADIUS. Its unit is Earth radii.
- pl_bmassprov is a categorical field describing mass provenance.
- Preserve the value "Msini" exactly. Do not expand it or describe it as stellar mass.
- Do not invent interpretations of categorical fields.

Explain:
- Predicted screening label and label-match score.
- The five supplied input features.
- Scientific limitations.

Evidence:
{json.dumps(evidence, indent=2, allow_nan=False)}

Write a concise explanation for a student project."""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_ctx": 2048,
            "num_predict": 400,
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach local Ollama server.") from exc

    if not result.get("done"):
        raise RuntimeError("Qwen generation did not complete.")

    return result.get("response", "").strip()

