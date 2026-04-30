import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.data import load_sentiment_dataset, tokenize_dataset
from src.model import compute_metrics, MODEL_NAME


def prepare_dataset(tokenizer, split, max_samples=None):
    dataset = load_sentiment_dataset(split=split, max_samples=max_samples)
    dataset = tokenize_dataset(dataset, tokenizer)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ["input_ids", "attention_mask", "label"]])
    return dataset.with_format("torch")


def train_model(output_dir: str, epochs: int = 1, max_train_samples: int | None = None, max_eval_samples: int | None = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(tokenizer, "train", max_train_samples)
    eval_dataset = prepare_dataset(tokenizer, "validation", max_eval_samples)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model with Hugging Face Transformers.")
    parser.add_argument("--output-dir", default="outputs/sentiment-model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-train-samples", type=int, default=1024, help="Limit for train split size for demo and CI.")
    parser.add_argument("--max-eval-samples", type=int, default=256, help="Limit for validation split size for demo and CI.")
    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
