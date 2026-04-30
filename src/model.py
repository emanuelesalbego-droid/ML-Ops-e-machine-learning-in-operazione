import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]


def load_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = MODEL_NAME):
    return AutoModelForSequenceClassification.from_pretrained(model_name)


def sentiment_pipeline(model_name: str = MODEL_NAME, device: int = -1) -> Pipeline:
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)


def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    references = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="weighted")
    accuracy = accuracy_score(references, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
