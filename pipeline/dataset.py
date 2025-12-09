from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import KFold

def load_data(dataset_cfg):
    """
    Load dataset and optionally perform cross-validation splitting.
    Returns either a single Dataset or a dict of DatasetDicts.
    """
    dataset_path = dataset_cfg.get("path", "./data/medical.jsonl")
    split_name = dataset_cfg.get("split", "train")

    # Load local JSONL dataset
    ds = load_dataset("json", data_files=dataset_path)[split_name]

    cv_splits = dataset_cfg.get("cross_validation", False)
    n_splits = dataset_cfg.get("n_splits", 5)

    if cv_splits:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = {}
        texts = ds["text"]
        indices = list(range(len(texts)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]

            train_ds = Dataset.from_dict({"text": train_texts})
            val_ds = Dataset.from_dict({"text": val_texts})

            splits[f"fold_{fold}"] = DatasetDict({
                "train": train_ds,
                "validation": val_ds
            })
        return splits

    return ds
