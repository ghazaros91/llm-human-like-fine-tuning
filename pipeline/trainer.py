from pathlib import Path
from pipeline.finetune import load_base_model, apply_lora, finetune_supervised
from pipeline.reinforcement import apply_rlhf
from pipeline.adversarial import apply_adversarial
from pipeline.dataset import load_data

def train_pipeline(cfg):
    """
    Unified training pipeline supporting:
    - Ollama inference (no LoRA/QLoRA training)
    - RLHF (emulated)
    - Adversarial inference
    """
    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    finetune_cfg = cfg.get("finetune", {})
    reinforcement_cfg = cfg.get("reinforcement", {})
    adversarial_cfg = cfg.get("adversarial", {})

    dataset_path = dataset_cfg.get("path", "./data/medical.jsonl")
    dataset = load_data(dataset_cfg)

    model = load_base_model(model_cfg, finetune_cfg)

    # LoRA / QLoRA skipped for Ollama
    finetuning_method = train_cfg.get("finetuning_method", "").lower()
    if finetuning_method in ["lora", "qlora"]:
        model = apply_lora(model, finetune_cfg)

    # Step 1: RLHF
    if reinforcement_cfg.get("enabled", False):
        apply_rlhf(model, dataset_path, reinforcement_cfg)

    # Step 2: Adversarial
    if adversarial_cfg.get("enabled", False):
        generator_model = adversarial_cfg.get("generator_model", model_cfg.get("model_name", model_cfg.get("model", "")))
        discriminator_model = adversarial_cfg.get("discriminator_model", "")
        apply_adversarial(generator_model, discriminator_model, dataset_path, adversarial_cfg)

    # Step 3: Supervised inference
    finetune_supervised(model, dataset_path, finetune_cfg)

    output_dir = Path(train_cfg.get("output_dir", "./outputs"))
    print(f"Training pipeline completed. Outputs saved to {output_dir}")
