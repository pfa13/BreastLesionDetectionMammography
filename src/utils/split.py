from sklearn.model_selection import train_test_split
from pathlib import Path

def create_splits(images, train_ratio=0.7, val_ratio=0.15):
    train, temp = train_test_split(
        images,
        test_size=(1 - train_ratio),
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42
    )

    return train, val, test


def save_splits(splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, files in splits.items():
        with open(output_dir / f"{name}.txt", "w") as f:
            for path in files:
                f.write(path + "\n")

    print("[OK] Splits saved")