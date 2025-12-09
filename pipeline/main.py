from pipeline.config_loader import load_config
from pipeline.trainer import train_pipeline
from pathlib import Path

def main():
    config_path = Path("configs/pipeline.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(str(config_path))
    train_pipeline(cfg)

if __name__ == "__main__":
    main()
