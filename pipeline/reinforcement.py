import json
import random
import ollama
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def safe_generate(model, prompt):
    try:
        resp = ollama.generate(model=model, prompt=prompt, stream=False)
        return resp.get("response", "")
    except Exception as e:
        logger.error(f"Ollama RLHF generation failed: {e}")
        return ""


def apply_rlhf(model_name: str, dataset_path: str, cfg: dict):
    if not cfg.get("enabled", False):
        logger.info("RLHF disabled. Skipping.")
        return

    logger.info(f"RLHF emulation enabled for Ollama model: {model_name}")

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    for item in dataset[:cfg.get("num_samples", 32)]:
        prompt = item.get("text", "")
        _ = safe_generate(model_name, prompt)

    logger.info("RLHF emulation complete.")
