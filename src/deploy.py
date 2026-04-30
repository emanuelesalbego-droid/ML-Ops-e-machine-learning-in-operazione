import argparse
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def deploy_to_hub(model_dir: str, repo_id: str, token: str):
    api = HfApi()
    api.create_repo(repo_id=repo_id, token=token, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.push_to_hub(repo_id, use_auth_token=token)
    tokenizer.push_to_hub(repo_id, use_auth_token=token)
    print(f"Model deployed to Hugging Face Hub: {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a fine-tuned sentiment model to Hugging Face Hub.")
    parser.add_argument("--model-dir", default="outputs/sentiment-model", help="Local model directory to push.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/model-name.")
    parser.add_argument("--token", required=True, help="Hugging Face access token.")
    args = parser.parse_args()

    deploy_to_hub(args.model_dir, args.repo_id, args.token)
