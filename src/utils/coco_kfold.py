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

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(image_ids)):
        train_ids = set([image_ids[i] for i in train_idx])
        val_ids = set([image_ids[i] for i in val_idx])

        train_images = [img for img in images if img["id"] in train_ids]
        val_images = [img for img in images if img["id"] in val_ids]

        train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
        val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids]

        folds.append({
            "train": {
                "images": train_images,
                "annotations": train_annotations,
                "categories": categories
            },
            "val": {
                "images": val_images,
                "annotations": val_annotations,
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
                json.dump(fold[split], f)

    print("[OK] COCO folds saved")