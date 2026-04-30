# Libreria per caricamento dataset pubblici da Hugging Face Dataset Hub
from datasets import load_dataset

# Nome e configurazione del dataset pubblico utilizzato: tweet_eval con task 'sentiment'
# Contiene ~22k tweet etichettati manualmente con sentiment (negative, neutral, positive)
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"


def load_sentiment_dataset(split: str = "train", max_samples: int | None = None):
    """Carica il dataset tweet_eval dalla cache locale o da HF Hub.
    
    Arrivato qui da:
    - Utilizza la libreria Hugging Face Datasets per gestire download e cache automatici
    - max_samples serve per limitare i dati nei test e nella CI/CD (velocizza l'esecuzione)
    
    Perché è qui:
    - Centralizza la logica di caricamento dati in un'unica funzione
    - Facilita le future modifiche (ad es. loading di dataset diversi)
    - Permette limiti di campioni per demo senza perdere la struttura del dataset
    """
    # Carica il dataset da Hugging Face Hub (se non già cached localmente)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    # Se è specificato un limite, taglia il dataset al numero di campioni richiesto
    if max_samples is not None:
        limit = min(max_samples, len(dataset))
        dataset = dataset.select(range(limit))
    return dataset


def tokenize_dataset(dataset, tokenizer, text_column: str = "text", max_length: int = 128):
    """Tokenizza un dataset convertendo i testi in token e aggiungendo padding.
    
    Arrivato qui da:
    - Utilizza il metodo .map() della libreria Datasets per applicare la trasformazione in batch
    - Questo è più efficiente che tokenizzare uno alla volta
    - Il padding garantisce che tutti i testi abbiano la stessa lunghezza (128 token)
    
    Perché è qui:
    - Separa la logica di preprocessing dal caricamento dati
    - Tokenizzazione è necessaria prima di passare i testi al modello.
    - max_length=128 è un compromesso tra lunghezza media dei tweet e memoria
    """
    # Applica tokenizzazione in batch: converte testo-&gt; token IDs + attention masks
    return dataset.map(
        lambda batch: tokenizer(
            batch[text_column],
            truncation=True,  # Tronca testi più lunghi di max_length
            padding="max_length",  # Aggiunge padding per raggiungere max_length
            max_length=max_length
        ),
        batched=True,  # Elabora più campioni per volta (più veloce)
    )
