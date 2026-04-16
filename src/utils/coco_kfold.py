import json
from sklearn.model_selection import KFold
from pathlib import Path


def create_coco_kfold(annotation_file, k=10):
    with open(annotation_file) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    image_ids = [img["id"] for img in images]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    folds = []

    for train_idx, val_idx in kf.split(image_ids):
        train_ids = set([image_ids[i] for i in train_idx])
        val_ids = set([image_ids[i] for i in val_idx])

        folds.append({
            "train": {
                "images": [img for img in images if img["id"] in train_ids],
                "annotations": [ann for ann in annotations if ann["image_id"] in train_ids],
                "categories": categories
            },
            "val": {
                "images": [img for img in images if img["id"] in val_ids],
                "annotations": [ann for ann in annotations if ann["image_id"] in val_ids],
                "categories": categories
            }
        })

    return folds


def save_coco_folds(folds, output_dir):
    output_dir = Path(output_dir)

    for i, fold in enumerate(folds):
        fold_dir = output_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val"]:
            with open(fold_dir / f"{split}.json", "w") as f:
                json.dump(fold[split], f, indent=2)

    print("[OK] COCO folds saved")