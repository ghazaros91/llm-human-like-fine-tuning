import yaml
from pathlib import Path

def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_config(pipeline_config_path: str):
    """
    Loads pipeline configuration and merges dataset, model, finetune, reinforcement, adversarial configs.
    """
    train_cfg = load_yaml(pipeline_config_path).get("train", {})

    dataset_cfg_path = Path("configs/datasets") / f"{train_cfg['dataset']}.yml"
    model_cfg_path = Path("configs/models") / f"{train_cfg['model']}.yml"
    finetune_cfg_path = Path("configs/finetune") / f"{train_cfg['finetuning_method']}.yml"
    reinforcement_cfg_path = Path("configs/train/reinforcement.yml")
    adversarial_cfg_path = Path("configs/train/adversarial.yml")

    dataset_cfg = load_yaml(dataset_cfg_path)
    model_cfg = load_yaml(model_cfg_path)
    finetune_cfg = load_yaml(finetune_cfg_path)
    reinforcement_cfg = load_yaml(reinforcement_cfg_path).get("reinforcement", {})
    adversarial_cfg = load_yaml(adversarial_cfg_path).get("adversarial", {})

    return {
        "train": train_cfg,
        "dataset": dataset_cfg,
        "model": model_cfg,
        "finetune": finetune_cfg,
        "reinforcement": reinforcement_cfg,
        "adversarial": adversarial_cfg
    }
