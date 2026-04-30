import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
from sklearn.metrics import classification_report

from src.data import load_sentiment_dataset
from src.model import LABELS, sentiment_pipeline


def monitor_sentiment(sample_size: int = 256, output_path: str | None = None):
    # Carica il dataset di test limitato al numero campioni richiesto
    dataset = load_sentiment_dataset("test", max_samples=sample_size)
    # Crea il pipeline di sentiment analysis con il modello pre-addestrato
    sentiment = sentiment_pipeline()
    # Converti la colonna text del dataset in una lista di stringhe (necessario per il pipeline)
    texts = list(dataset["text"])
    labels = dataset["label"]
    # Esegui le predizioni sul batch di testi, estraendo solo l'etichetta da ogni risultato
    predictions = [prediction["label"] for prediction in sentiment(texts, truncation=True)]
    # Crea una mappa che associa le etichette di testo (negative, neutral, positive) ai loro indici numerici
    label_map = {label: idx for idx, label in enumerate(LABELS)}
    # Aggiungi anche il mapping specifico del modello (LABEL_0, LABEL_1, LABEL_2)
    label_map.update({f"LABEL_{idx}": idx for idx in range(len(LABELS))})
    # Converti le predizioni in indici numerici usando la mappa, default -1 per etichette sconosciute
    prediction_ids = [label_map.get(label, -1) for label in predictions]
    # Calcola la distribuzione dei sentiment predetti
    distribution = Counter(predictions)
    # Genera un report di classificazione dettagliato con precision, recall, f1-score
    report = classification_report(labels, prediction_ids, target_names=LABELS, output_dict=True, zero_division=0)
    # Aggrega i risultati in un dizionario riassuntivo
    summary = {
        "sample_size": len(texts),
        "distribution": dict(distribution),
        "report": report,
    }
    # Se specificato un path di output, salva il report in formato JSON
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
