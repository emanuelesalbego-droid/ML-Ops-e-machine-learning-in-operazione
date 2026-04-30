from datasets import load_dataset

DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"


def load_sentiment_dataset(split: str = "train", max_samples: int | None = None):
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def tokenize_dataset(dataset, tokenizer, text_column: str = "text", max_length: int = 128):
    return dataset.map(
        lambda batch: tokenizer(batch[text_column], truncation=True, padding="max_length", max_length=max_length),
        batched=True,
    )
