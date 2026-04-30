# Test suite per il modulo di prediction del sentiment
import pytest
from src.predict import predict_text


# Parametrizza il test con 3 diversi esempi di testo per validare la robustezza
# Arrivato qui da: best practice in pytest per testare più input senza duplicare codice
# Perché è qui: verifica che funzioni sia per sentiment positivi, neutrali che negativi
@pytest.mark.parametrize(
    "text",
    [
        "I love this product, it is amazing!",  # Sentiment positivo
        "This is okay, not great but not bad.",  # Sentiment neutrale
        "I hate it, the service was terrible.",  # Sentiment negativo
    ],
)
def test_sentiment_prediction_contains_label(text):
    """Test che la funzione predict_text ritorna un'etichetta sentiment valida.
    
    Arrivato qui da:
    - Unit test semplice per validare il pipeline di inference end-to-end
    - Usa predict_text() direttamente per testare come l'utente lo userebbe
    
    Perché è qui:
    - Assicura che il modello carica correttamente e può fare predizioni
    - Verifica che l'output ha la struttura attesa (lista con dict contenente label e score)
    - Supporta múltiple formati di label (minuscole, maiuscole, LABEL_X) per robustezza
    """
    # Chiama la funzione di prediction
    result = predict_text(text)
    # Verifica che il risultato sia una lista (il pipeline ritorna sempre una lista)
    assert isinstance(result, list)
    # Verifica che la lista abbia esattamente un elemento (una predizione per un testo)
    assert len(result) == 1
    # Verifica che il dizionario contenga la chiave 'label' (etichetta di sentiment)
    assert "label" in result[0]
    # Verifica che il dizionario contenga la chiave 'score' (confidenza della predizione)
    assert "score" in result[0]
    # Verifica che l'etichetta sia uno dei valori attesi (consente variazioni di caso)
    assert result[0]["label"] in {
        "positive",  # Formato minuscolo standard
        "negative",
        "neutral",
        "POSITIVE",  # Formato maiuscolo alternativo
        "NEGATIVE",
        "NEUTRAL",
        "LABEL_0",  # Formato numerico (indice classe) usato da alcuni modelli
        "LABEL_1",
        "LABEL_2",
    }
