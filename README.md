# ML-Ops e machine learning in operazione

Questo repository implementa un progetto ML-Ops per l'analisi del sentiment su testi social media.

## Cosa contiene

- `src/train.py`: training e fine-tuning di un modello di sentiment analysis su dataset pubblici.
- `src/predict.py`: inferenza su testo libero con il modello pre-addestrato.
- `src/deploy.py`: deploy opzionale del modello su Hugging Face Hub.
- `src/monitor.py`: script di monitoraggio per valutare performance e distribuzione del sentiment.
- `.github/workflows/ci.yml`: pipeline CI/CD per test, training dimostrativo e deploy condizionale.
- `notebooks/ML-Ops-sentiment-analysis.ipynb`: notebook di progetto con passaggi esecutivi e documentazione.

## Modello utilizzato

Viene utilizzato il modello pre-addestrato `cardiffnlp/twitter-roberta-base-sentiment-latest` per sentiment analysis su testi da social media.

## Come usare

1. Installare i pacchetti:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Eseguire il training di prova:
   ```bash
   python src/train.py --output-dir outputs/sentiment-model --epochs 1 --max-train-samples 512 --max-eval-samples 128
   ```
3. Fare inferenza su un testo:
   ```bash
   python src/predict.py --text "I love this feature!"
   ```
4. Eseguire i test automatici:
   ```bash
   pytest -q tests
   ```

## Note CI/CD

La pipeline GitHub Actions nella cartella `.github/workflows/ci.yml` esegue test automatici su ogni push e pull request. Il deploy verso Hugging Face Hub avviene solo su `workflow_dispatch` se è impostato il segreto `HF_TOKEN` e `HF_MODEL_REPO`.
