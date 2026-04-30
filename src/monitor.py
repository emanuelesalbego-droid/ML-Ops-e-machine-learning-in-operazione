import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from sklearn.metrics import classification_report

from src.data import load_sentiment_dataset
from src.model import LABELS, sentiment_pipeline


def monitor_sentiment(sample_size: int = 256, output_path: str | None = None):
    dataset = load_sentiment_dataset("test", max_samples=sample_size)
    pipeline = sentiment_pipeline()
    texts = dataset["text"]
    labels = dataset["label"]
    predictions = [prediction["label"] for prediction in pipeline(texts, truncation=True)]
    label_map = {label: idx for idx, label in enumerate(LABELS)}
    label_map.update({f"LABEL_{idx}": idx for idx in range(len(LABELS))})
    prediction_ids = [label_map.get(label, -1) for label in predictions]
    distribution = Counter(predictions)
    report = classification_report(labels, prediction_ids, target_names=LABELS, output_dict=True, zero_division=0)
    summary = {
        "sample_size": len(texts),
        "distribution": distribution,
        "report": report,
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a monitoring evaluation pass for sentiment performance.")
    parser.add_argument("--sample-size", type=int, default=128, help="Number of test examples to evaluate.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to store monitoring metrics.")
    args = parser.parse_args()
    result = monitor_sentiment(sample_size=args.sample_size, output_path=args.output)
    print(result)
