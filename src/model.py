# Librerie per caricamento modelli e valutazione delle metriche
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

# Identificativo del modello pre-addestrato per l'analisi del sentiment da testi social media
# Scelto perché specificamente fine-tuned su dati Twitter con 3 label: negative, neutral, positive
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# Elenco delle etichette di sentiment che il modello può predire
LABELS = ["negative", "neutral", "positive"]


def load_tokenizer(model_name: str = MODEL_NAME):
    """Carica il tokenizer associato al modello pre-addestrato.
    Necessario per convertire il testo in token prima di passarlo al modello.
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(model_name: str = MODEL_NAME):
    """Carica il modello di classificazione sequenziale da Hugging Face Hub.
    Utilizza un checkpoint pre-addestrato per evitare il training da zero.
    """
    return AutoModelForSequenceClassification.from_pretrained(model_name)


def sentiment_pipeline(model_name: str = MODEL_NAME) -> Pipeline:
    """Crea un pipeline di sentiment analysis che combina tokenizer e modello.
    Ritorna un oggetto Pipeline che si occupa automaticamente della tokenizzazione,
    forward pass del modello, e post-processing delle logit in probabilità.
    Device è impostato a CPU di default poiché l'ambiente ha solo CPU.
    """
    # Nota: Il parametro device è omesso perché il sistema ha solo CPU disponibile.
    # Il pipeline usa automaticamente la risorsa di calcolo disponibile (CPU in questo caso).
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


def compute_metrics(pred):
    """Calcola metriche di valutazione per il training del modello.
    Prende in input un oggetto EvalPrediction da Hugging Face Trainer
    e ritorna accuracy, precision, recall e f1-score pesati.
    
    Arrivato qui da:
    - Durante il training, il Trainer chiama questa funzione ad ogni epoca
    - Utilizza np.argmax per convertire logit in label predetti
    - Usa average='weighted' per gestire classi sbilanciate
    """
    # Estrai le predizioni dal formato raw (logit) convertendole in indici di label
    predictions = np.argmax(pred.predictions, axis=1)
    # Ottieni i label reali dal dataset
    references = pred.label_ids
    # Calcola precision, recall e f1 con media pesata (utile per dataset non bilanciato)
    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average="weighted")
    # Calcola l'accuracy (percentuale di predizioni corrette)
    accuracy = accuracy_score(references, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
