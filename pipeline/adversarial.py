import json
import random
import ollama
from pathlib import Path
from typing import Dict
import logging

# -----------------------
# Logging Configuration
# -----------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def safe_generate(model: str, prompt: str) -> str:
    """
    Safe wrapper for Ollama generation.
    Forces stream=False so the models don't block.
    """
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False  # IMPORTANT FIX
        )
        return resp.get("response", "")
    except Exception as e:
        logger.error(f"Ollama generation failed for model={model}: {e}")
        return ""


def load_generator_discriminator(generator_cfg: Dict, discriminator_cfg: Dict):
    """
    Load generator and discriminator model names.
    """
    generator_model = generator_cfg["model_name"]
    discriminator_model = discriminator_cfg["model_name"]
    logger.info(f"Generator: {generator_model}, Discriminator: {discriminator_model}")
    return generator_model, discriminator_model


def apply_adversarial(generator_model: str, discriminator_model: str, dataset_path: str, adversarial_cfg: Dict):
    """
    Perform adversarial inference loop using two Ollama models.
    This is not true training — it only simulates adversarial reward generation.
    """
    if not adversarial_cfg.get("enabled", False):
        logger.info("Adversarial training not enabled. Skipping.")
        return

    batch_size = adversarial_cfg.get("batch_size", 4)
    epochs = adversarial_cfg.get("epochs", 3)
    output_dir = Path(adversarial_cfg.get("output_dir", "./adversarial_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Adversarial inference loop enabled. Generator: {generator_model}, Discriminator: {discriminator_model}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    for epoch in range(epochs):
        random.shuffle(dataset)
        logger.info(f"--- Starting epoch {epoch+1}/{epochs} ---")

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            prompts = [item.get("text", "") for item in batch]

            # ---- Generator Forward Pass ----
            gen_outputs = []
            for prompt in prompts:
                gen_text = safe_generate(generator_model, prompt)
                gen_outputs.append(gen_text)

                logger.debug(f"[GEN] prompt: {prompt[:40]}... -> {gen_text[:80]}...")

            # ---- Discriminator Reward Scoring ----
            rewards = []
            for output in gen_outputs:
                reward_raw = safe_generate(discriminator_model, prompt)
                try:
                    reward = float(reward_raw.strip())
                except ValueError:
                    reward = 0.5  # fallback score
                rewards.append(reward)

                logger.debug(f"[DISC] output: {output[:40]}... -> reward={reward}")

            # ---- Save outputs ----
            batch_file = output_dir / f"epoch{epoch+1}_batch{i//batch_size+1}.jsonl"
            with open(batch_file, "w") as f_out:
                for prompt, output, reward in zip(prompts, gen_outputs, rewards):
                    json.dump({
                        "prompt": prompt,
                        "generated": output,
                        "reward": reward
                    }, f_out)
                    f_out.write("\n")

            logger.info(f"Saved batch {i//batch_size+1} for epoch {epoch+1}: {batch_file}")

    logger.info(f"Adversarial inference complete. Results saved to: {output_dir}")
