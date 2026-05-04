# ML-Ops e machine learning in operazione

Questo repository implementa un progetto ML-Ops per l'analisi del sentiment su testi social media.

## Cosa contiene

- `src/train.py`:  training/fine-tuning del modello.
- `src/predict.py`: inferenza su testo.
- `src/deploy.py`: deploy su Hugging Face Hub.
- `src/monitor.py`: valutazione/monitoraggio.

- `data.py` — gestione dei dati
- `model.py` — definizione del modello
- `test_model.py` — test automatici del codice

- `.github/workflows/ci.yml`: pipeline CI/CD per test, package install, training dimostrativo e deploy condizionale.
- `pyproject.toml`: metadata e configurazione per installazione del pacchetto.
- `notebooks/ML-Ops-sentiment-analysis.ipynb`: notebook documentazione con setup, dataset, inferenza e note di deployment.

## Modello utilizzato

Viene utilizzato il modello pre-addestrato `cardiffnlp/twitter-roberta-base-sentiment-latest` per sentiment analysis su testi da social media.

## Preparazione ambiente
1. Creazione e attivazione un ambiente virtuale:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
## Installazione

1. Installare il pacchetto PIP al'ultima verisone ed installo il progetto in modalità di sviluppo:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e .
   ```
2. Installare le dipendenze aggiuntive, incluso il pacchetto PyTorch CPU:
   ```bash
   python -m pip install -r requirements.txt
   ```

> Se utilizzi un ambiente con GPU, sostituisci la riga `torch==2.2.2+cpu` in `requirements.txt` con la versione PyTorch GPU appropriata.

## Come usare

1. Eseguire il training di prova:
   ```bash
   python src/train.py --output-dir outputs/sentiment-model --epochs 1 --max-train-samples 512 --max-eval-samples 128
   ```
3. controllare che venga creato:
   ```
   outputs/sentiment-model
   ```
4. Testo inferenza su un testo:
   ```bash
   python src/predict.py --text "I love this feature!"
   ```
5. Controllo che lo script restituisca una classe di sentiment e un punteggio.
6. Eseguire il monitoraggio:
   ```bash
   python src/monitor.py --sample-size 128 --output monitor-report.json
   ```
7. Esamino il file monitor-report.json o l’output sulla console
4. Eseguire i test automatici:
   ```bash
   pytest -q tests
   ```

## Deploy su Hugging Face Hub

Per eseguire il deploy manuale:
1. Imposto i segreti `HF_TOKEN` e `HF_MODEL_REPO`
2. Quindi uso il workflow GitHub Actions con `workflow_dispatch` o eseguo:

```bash
python src/deploy.py --model-dir outputs/sentiment-model --repo-id username/model-name --token $HF_TOKEN
```

## Note CI/CD

La pipeline GitHub Actions esegue ora i seguenti passaggi:
- installazione del pacchetto con `pip install -e .`
- installazione delle dipendenze da `requirements.txt`
- esecuzione di `pytest`
- training dimostrativo e deploy condizionale su Hugging Face Hub se configurato tramite segreti.

## Notebook

Apri `notebooks/ML-Ops-sentiment-analysis.ipynb` per una guida passo-passo su come usare il dataset, eseguire inferenza e comprendere le aree di monitoraggio e deploy.