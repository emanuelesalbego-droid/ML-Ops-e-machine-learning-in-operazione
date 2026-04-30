# Script di training/fine-tuning del modello di sentiment analysis
import argparse
import os
import sys

# Aggiunge la cartella root al path di Python per permettere import di moduli src
# Necessario quando lo script viene eseguito direttamente da CLI
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
    """Prepara il dataset per il training rimuovendo colonne non necessarie e tokenizzando.
    
    Arrivato qui da:
    - Il Trainer di HF richiede dataset in formato torch con solo input_ids, attention_mask, label
    - Altre colonne (es. tweet ID, URL) causano errori se lasciate
    
    Perché è qui:
    - Centralizza la preparazione del dataset (load -> tokenize -> clean -> format)
    - Riutilizzabile per train e validation split
    """
    # Carica il dataset dal source (local cache o HF Hub)
    dataset = load_sentiment_dataset(split=split, max_samples=max_samples)
    # Tokenizza i testi convertendoli in token IDs e aggiungendo padding
    dataset = tokenize_dataset(dataset, tokenizer)
    # Rimuove tutte le colonne tranne quelle richieste dal Trainer di HF
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    # Converte il dataset al formato PyTorch (necessario per il Trainer)
    return dataset.with_format("torch")


def train_model(output_dir: str, epochs: int = 1, max_train_samples: int | None = None, max_eval_samples: int | None = None):
    """Fine-tuning del modello pre-addestrato sul dataset tweet_eval.
    
    Arrivato qui da:
    - Utilizza l'API Trainer di Hugging Face (semplifica configurazione e distribuzione)
    - Learning rate 2e-5 è standard per fine-tuning di modelli BERT (non da zero)
    - Batch size e parametri sono tuned per balancio memoria/accuratezza
    
    Perché è qui:
    - Centralizza il training loop in una funzione riutilizzabile
    - Permette parametrizzazione (epochs, sample limits per CI/CD)
    - Salva il modello fine-tuned per deploy
    """
    # Carica tokenizer e modello pre-addestrati
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Prepara train e validation dataset
    train_dataset = prepare_dataset(tokenizer, "train", max_train_samples)
    eval_dataset = prepare_dataset(tokenizer, "validation", max_eval_samples)

    # Data collator gestisce il padding dinamico durante il training (più efficiente)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Configura gli iperparametri di training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Valuta a fine di ogni epoca
        save_strategy="epoch",  # Salva checkpoint a fine di ogni epoca
        learning_rate=2e-5,  # Learning rate conservativo per fine-tuning
        per_device_train_batch_size=8,  # Batch size per GPU/CPU (ridotto per CPU)
        per_device_eval_batch_size=16,  # Batch size di valutazione (non aggiorna pesi)
        num_train_epochs=epochs,  # Numero di epoche da trainare
        weight_decay=0.01,  # L2 regularization per evitare overfitting
        logging_steps=50,  # Loga metriche ogni 50 step
        save_total_limit=2,  # Mantiene solo gli ultimi 2 checkpoint
        load_best_model_at_end=True,  # Carica il modello migliore alla fine
        metric_for_best_model="accuracy",  # Metrica per selezionare il miglior checkpoint
    )

    # Crea l'oggetto Trainer che coordina training, valutazione e salvataggio
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Esegue il training (fino a 'epochs' epoche)
    trainer.train()
    # Salva il modello fine-tuned nella cartella output_dir
    trainer.save_model(output_dir)
    # Salva il tokenizer (necessario per inference)
    tokenizer.save_pretrained(output_dir)
    return trainer


if __name__ == "__main__":
    # Parser per accettare parametri da CLI
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model with Hugging Face Transformers.")
    parser.add_argument("--output-dir", default="outputs/sentiment-model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-train-samples", type=int, default=1024, help="Limit for train split size for demo and CI.")
    parser.add_argument("--max-eval-samples", type=int, default=256, help="Limit for validation split size for demo and CI.")
    args = parser.parse_args()

    # Avvia il training del modello con i parametri specificati
    train_model(
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
