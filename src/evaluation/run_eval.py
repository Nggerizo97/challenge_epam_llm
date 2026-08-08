from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.ragas_eval import run_ragas_evaluation


if __name__ == "__main__":
    samples_path = Path("data/eval/samples.json")
    if not samples_path.exists():
        raise FileNotFoundError(
            "Missing data/eval/samples.json. Add evaluation samples before running."
        )

    with samples_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    scores = run_ragas_evaluation(samples)
    print("RAG Evaluation Scores")
    for metric_name, metric_value in scores.items():
        print(f"- {metric_name}: {metric_value:.4f}")
