import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import pipeline
from src.model import MODEL_NAME


def predict_text(text: str, model_name: str = MODEL_NAME):
    sentiment = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return sentiment(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment prediction on social-media text.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Pretrained model or local path.")
    args = parser.parse_args()

    result = predict_text(args.text, model_name=args.model_name)
    print(result)
