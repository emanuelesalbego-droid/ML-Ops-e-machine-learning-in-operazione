# Script di inferenza per classificazione del sentiment su testo libero
import argparse
import os
import sys

# Aggiunge la cartella root al path di Python per permettere import di moduli src
# Necessario quando lo script viene eseguito direttamente (non come modulo installato)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import pipeline
from src.model import MODEL_NAME


def predict_text(text: str, model_name: str = MODEL_NAME):
    """Esegue la classificazione di sentiment su un testo singolo.
    
    Arrivato qui da:
    - Wrapper attorno al pipeline di Hugging Face
    - Centralizza la logica di inferenza per riutilizzo nei test e altrove
    
    Perché è qui:
    - Consente di cambiare il modello facilmente (parametro model_name)
    - Separa la logica di inferenza dall'CLI
    - Più facile da testare unitariamente
    """
    # Crea il pipeline di sentiment analysis caricando modello e tokenizer
    sentiment = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    # Esegue la predizione e ritorna una lista con label e score
    return sentiment(text)


if __name__ == "__main__":
    # Configura l'argument parser per accettare testo e modello da CLI
    parser = argparse.ArgumentParser(description="Run sentiment prediction on social-media text.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Pretrained model or local path.")
    args = parser.parse_args()

    # Esegue la predizione e stampa il risultato
    result = predict_text(args.text, model_name=args.model_name)
    print(result)
