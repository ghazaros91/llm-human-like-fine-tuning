import json
import ollama
from typing import Dict

def apply_rlhf(model_name: str, dataset_path: str, rlhf_cfg: Dict):
    """
    Emulated RLHF step using Ollama (inference-only).
    Produces output and prints "reward" placeholder.
    """
    if not rlhf_cfg.get("enabled", False):
        return

    batch_size = rlhf_cfg.get("batch_size", 4)

    print(f"RLHF emulation enabled for Ollama model: {model_name}")

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        for example in batch:
            prompt = example.get("text", "")
            if not prompt:
                continue

            resp = ollama.generate(model=model_name, prompt=prompt)
            output = resp.get("response", "")
            # Emulate reward signal
            reward = 0.5
            print(f"Prompt: {prompt}\nGenerated: {output}\nReward: {reward}\n")
