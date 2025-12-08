from pipeline.config_loader import load_config
from pipeline.trainer import train_pipeline

def main():
    cfg = load_config("configs/pipeline.yml")
    train_pipeline(cfg)

if __name__ == "__main__":
    main()

