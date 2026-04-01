from sklearn.model_selection import KFold
from pathlib import Path

def create_kfold_splits(images, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    folds = []

    images = list(images)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images)):
        train = [images[i] for i in train_idx]
        val = [images[i] for i in val_idx]

        folds.append({
            "fold": fold_idx,
            "train": train,
            "val": val
        })

    return folds

def save_kfold_splits(folds, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        fold_dir = output_dir / f"fold_{fold['fold']}"
        fold_dir.mkdir(exist_ok=True)

        with open(fold_dir / "train.txt", "w") as f:
            for path in fold["train"]:
                f.write(path + "\n")

        with open(fold_dir / "val.txt", "w") as f:
            for path in fold["val"]:
                f.write(path + "\n")

    print("[OK] K-Fold splits saved")