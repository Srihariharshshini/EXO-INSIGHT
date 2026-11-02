# EXO-INSIGHT — AI-use session notes

These notes summarize AI-assisted work sessions. They are not conversation exports. The original conversation exports must be added separately before submission.

## Session: M6 local model integration

Kept:
- Python integration with local Ollama.
- Evidence-grounded Qwen explanation structure.
- Strict separation between classifier output and natural-language explanation.

Fixed or reviewed:
- Python import paths.
- Qwen output constraints.
- Explanation token limit was identified as a source of truncation; a fully complete explanation still needs final validation.

Prompt changes:
- Added instructions not to invent feature definitions.
- Required the score to be described as a label-match score.
- Prohibited habitability or life claims.

## Session: M7 measured comparison

Kept:
- Ten balanced held-out benchmark tasks.
- Three Qwen-only repetitions per task.
- Saved raw responses and benchmark summary.

Reviewed:
- Qwen predicted label 1 for all ten tasks.
- Both classifier and Qwen failure traces were inspected.
- Results were described as exploratory due to the small sample.

## Session: M8 automated tests

Kept:
- Existing six tests.
- Added tests for threshold decisions, invalid names, duplicate names and missing model artifacts.

Fixed:
- Pytest initially was unavailable in the active Python environment.
- Added pytest to the environment.
- Added TESTS/conftest.py to make SRC importable.
- Corrected the fake model fixture to return a NumPy array instead of a Python list.

Verified:
- 11 automated tests passed.

## Session: M8 uncertainty analysis and documentation

Kept:
- Bootstrap confidence interval approach.
- 2,000 bootstrap resamples with random seed 42.
- Explicit limitations concerning pseudo-labels and sample size.

Created:
- README.md
- EVIDENCE.md
- requirements.txt
- This AI-use session summary.

## Required final action

Export the relevant AI conversations and place the exports in this folder. Review the notes and correct any details that do not accurately reflect the student's own work.
