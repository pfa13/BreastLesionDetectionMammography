import argparse
from pathlib import Path

from extract import extract_zip
from split import create_kfold_splits, save_kfold_splits
from coco_converter import convert_dmid_to_coco
from file_utils import collect_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al .zip")
    args = parser.parse_args()

    raw_dir = Path("../../data/raw")
    splits_dir = Path("../../data/splits")
    annotations_dir = Path("../../data/annotations")

    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract
    extract_zip(args.input, raw_dir)

    # 2. Search images
    images = collect_images(raw_dir)
    print(f"[INFO] {len(images)} imágenes encontradas")

    # 3. Split (K-Fold)
    folds = create_kfold_splits(images, k=10)
    save_kfold_splits(folds, splits_dir)

    # 4. COCO annotations por fold
    metadata_path = raw_dir / "Metadata.xlsx"

    for fold in folds:        
        fold_id = fold["fold"]
        print(f"[OK] Fold {fold_id} annotations initiated")
        train_images = fold["train"]
        val_images = fold["val"]

        fold_ann_dir = annotations_dir / f"fold_{fold_id}"
        fold_ann_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Fold {fold_id} coco initiated")
        convert_dmid_to_coco(
            train_images,
            metadata_path,
            fold_ann_dir / "train.json"
        )

        convert_dmid_to_coco(
            val_images,
            metadata_path,
            fold_ann_dir / "val.json"
        )

        print(f"[OK] Fold {fold_id} annotations created")


if __name__ == "__main__":
    main()