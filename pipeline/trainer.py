import logging
from pathlib import Path
from pipeline.finetune import load_base_model, apply_lora, finetune_supervised
from pipeline.reinforcement import apply_rlhf
from pipeline.adversarial import apply_adversarial, load_generator_discriminator
from pipeline.dataset import load_data


def setup_logger(log_file: str = "training.log"):
    """Set up a logger for the training pipeline."""
    logger = logging.getLogger("TrainPipeline")
    logger.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def train_pipeline(cfg):
    logger = setup_logger(cfg.get("train", {}).get("log_file", "training.log"))
    logger.info("Starting training pipeline...")

    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    finetune_cfg = cfg.get("finetune", {})
    reinforcement_cfg = cfg.get("reinforcement", {})
    adversarial_cfg = cfg.get("adversarial", {})

    try:
        dataset_path = dataset_cfg.get("path")
        logger.info(f"Loading dataset from {dataset_path}...")
        dataset = load_data(dataset_cfg)
        logger.info(f"Dataset loaded successfully. Samples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    try:
        logger.info("Loading base model...")
        model = load_base_model(model_cfg, finetune_cfg)
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    try:
        logger.info("Starting supervised fine-tuning...")
        finetune_supervised(model, dataset_path, finetune_cfg)
        logger.info("Supervised fine-tuning completed.")
    except Exception as e:
        logger.error(f"Supervised fine-tuning failed: {e}")

    if reinforcement_cfg.get("enabled", False):
        try:
            logger.info("Applying RLHF...")
            apply_rlhf(model, dataset_path, reinforcement_cfg)
            logger.info("RLHF completed successfully.")
        except Exception as e:
            logger.error(f"RLHF failed: {e}")

    if adversarial_cfg.get("enabled", False):
        try:
            logger.info("Loading generator and discriminator models...")
            gen, disc = load_generator_discriminator(
                {"model_name": adversarial_cfg.get("generator_model")},
                {"model_name": adversarial_cfg.get("discriminator_model")},
            )
            logger.info("Applying adversarial training...")
            apply_adversarial(gen, disc, dataset_path, adversarial_cfg)
            logger.info("Adversarial training completed successfully.")
        except Exception as e:
            logger.error(f"Adversarial training failed: {e}")

    logger.info("Training pipeline completed successfully.")
    return model
