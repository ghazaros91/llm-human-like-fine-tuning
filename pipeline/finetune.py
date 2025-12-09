import json
from pathlib import Path
import ollama
from typing import Dict

def load_base_model(model_cfg: Dict, finetune_cfg: Dict):
    """
    Load base model using Ollama API.
    Ollama only supports inference; no LoRA/QLoRA training.
    """
    model_name = model_cfg.get("model_name") or model_cfg.get("model")
    if not model_name:
        raise KeyError("model_name missing in model config")

    print(f"Ollama will use model: {model_name} (inference only)")
    return model_name

def apply_lora(model_name: str, finetune_cfg: Dict):
    """
    Ollama does not support LoRA/QLoRA fine-tuning.
    """
    if "lora" in finetune_cfg or "qlora" in finetune_cfg:
        print("Warning: Ollama does not support LoRA/QLoRA fine-tuning. Skipping adapter application.")
    return model_name

def finetune_supervised(model_name: str, dataset_path: str, finetune_cfg: Dict):
    """
    Supervised inference-only using Ollama.
    Iterates over dataset and prints generated responses.
    """
    output_dir = Path(finetune_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = finetune_cfg.get("batch_size", 4)

    print(f"Running inference on dataset: {dataset_path} using Ollama model: {model_name}")

    with open(dataset_path, "r") as f:
        lines = [json.loads(line) for line in f]

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        for example in batch:
            prompt = example.get("text", "")
            if not prompt:
                continue

            resp = ollama.generate(model=model_name, prompt=prompt)
            generated_text = resp.get("response", "")

            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}\n")

    print("Ollama-based supervised inference complete.")
