# Script di deploy del modello fine-tuned su Hugging Face Hub
import argparse
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def deploy_to_hub(model_dir: str, repo_id: str, token: str):
    """Carica il modello fine-tuned su Hugging Face Hub per condivisione e deploy.
    
    Arrivato qui da:
    - HfApi fornisce API Python per creare repo e caricare file su HF Hub
    - Usare Hub permette a chiunque di scaricare e usare il modello
    - Il token autentifica la richiesta per repo privati
    
    Perché è qui:
    - Centralizza la logica di deploy (riusabile da workflow CI/CD)
    - Separa il deploy dal training
    - Permette deploy condizionale (solo quando secrets sono configurati)
    """
    # Inizializza l'API di Hugging Face Hub
    api = HfApi()
    # Crea il repository (o non fa nulla se esiste già con exist_ok=True)
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)

    # Carica il modello e tokenizer dal disco locale
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Carica il modello e tokenizer su Hub
    model.push_to_hub(repo_id, use_auth_token=token)
    tokenizer.push_to_hub(repo_id, use_auth_token=token)
    print(f"Model deployed to Hugging Face Hub: {repo_id}")


if __name__ == "__main__":
    # Parser per accettare parametri da CLI
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned sentiment model to Hugging Face Hub.")
    parser.add_argument("--model-dir", default="outputs/sentiment-model", help="Local model directory to push.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/model-name.")
    parser.add_argument("--token", required=True, help="Hugging Face access token.")
    args = parser.parse_args()

    # Esegue il deploy con i parametri forniti
    deploy_to_hub(args.model_dir, args.repo_id, args.token)
