import json
from pathlib import Path
import ollama
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_base_model(model_cfg: Dict, finetune_cfg: Dict):
    """
    Loads an Ollama model (base or LoRA-adapted).
    """
    model_name = model_cfg.get("model_name") or model_cfg.get("model")
    if not model_name:
        raise ValueError("Model name missing in config")

    logger.info(f"Ollama will use model: {model_name}")
    return model_name


def apply_lora(model_name: str, finetune_cfg: Dict):
    """
    Applies LoRA adapter by referencing an Ollama model
    whose Modelfile already includes an ADAPTER directive.
    """
    lora_path = finetune_cfg.get("lora_path")

    if not lora_path:
        logger.info("No LoRA specified. Using base model.")
        return model_name

    logger.info(f"Using existing LoRA adapter at: {lora_path}")
    logger.info("NOTE: Ollama cannot train LoRA—this assumes the adapter was created externally.")

    # The model must already be created via: `ollama create mymodel -f Modelfile`
    adapted_model = finetune_cfg.get("lora_model_name", f"{model_name}-lora")

    logger.info(f"Ollama will use LoRA-adapted model: {adapted_model}")
    return adapted_model


def safe_generate(model: str, prompt: str, seq_len=1024) -> str:
    try:
        resp = ollama.generate(model=model, prompt=prompt, stream=False)
        return resp.get("response", "")
    except Exception as e:
        logger.error(f"Ollama generation error: {e}")
        return ""


def finetune_supervised(model_name: str, dataset_path: str, finetune_cfg: Dict):
    """
    Supervised inference-only pipeline using Ollama + LoRA.
    """
    output_dir = Path(finetune_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)


    logger.info(f"Running supervised inference using model: {model_name}")

    with open(dataset_path, "r") as f:
        lines = [json.loads(line) for line in f]
        print('dataset loaded')

    for example in lines:
        prompt = example.get("instruction", "")
        resp = safe_generate(model_name, prompt)

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {resp}")
        

    logger.info("Inference complete.")
