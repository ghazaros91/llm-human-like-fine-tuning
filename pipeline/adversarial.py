import json
import random
import ollama
from pathlib import Path
from typing import Dict

def load_generator_discriminator(generator_cfg: Dict, discriminator_cfg: Dict):
    generator_model = generator_cfg["model_name"]
    discriminator_model = discriminator_cfg["model_name"]
    print(f"Generator: {generator_model}, Discriminator: {discriminator_model}")
    return generator_model, discriminator_model

def apply_adversarial(generator_model: str, discriminator_model: str, dataset_path: str, adversarial_cfg: Dict):
    if not adversarial_cfg.get("enabled", False):
        return

    batch_size = adversarial_cfg.get("batch_size", 4)
    epochs = adversarial_cfg.get("epochs", 3)
    output_dir = Path(adversarial_cfg.get("output_dir", "./adversarial_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Adversarial inference loop enabled. Generator: {generator_model}, Discriminator: {discriminator_model}")

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    for epoch in range(epochs):
        random.shuffle(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [item.get("text", "") for item in batch]

            gen_outputs = []
            for prompt in prompts:
                resp = ollama.generate(model=generator_model, prompt=prompt)
                gen_outputs.append(resp.get("response", ""))

            rewards = []
            for output in gen_outputs:
                resp = ollama.generate(model=discriminator_model, prompt=output)
                try:
                    reward = float(resp.get("response", "0.5").strip())
                except ValueError:
                    reward = 0.5
                rewards.append(reward)

            batch_file = output_dir / f"epoch{epoch+1}_batch{i//batch_size+1}.jsonl"
            with open(batch_file, "w") as f_out:
                for prompt, output, reward in zip(prompts, gen_outputs, rewards):
                    json.dump({"prompt": prompt, "generated": output, "reward": reward}, f_out)
                    f_out.write("\n")

        print(f"Epoch {epoch+1}/{epochs} completed.")

    print("Adversarial inference complete. Results saved to:", output_dir)
