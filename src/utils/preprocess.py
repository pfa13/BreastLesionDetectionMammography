import argparse
from pathlib import Path

from utils.extract import extract_zip
from utils.split import create_splits, save_splits
from utils.coco_converter import convert_dmid_to_coco
from utils.file_utils import collect_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al .zip")
    args = parser.parse_args()

    raw_dir = "data/raw"
    splits_dir = "data/splits"
    annotations_dir = "data/annotations"

    # 1. Extract
    extract_zip(args.input, raw_dir)

    # 2. Search images
    images = collect_images(raw_dir)
    print(f"[INFO] {len(images)} imágenes encontradas")

    # 3. Split
    train, val, test = create_splits(images)

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    save_splits(splits, splits_dir)

    # 4. COCO
    metadata_path = "data/raw/Metadata.xlsx"

    convert_dmid_to_coco(train, metadata_path, "data/annotations/train.json")
    convert_dmid_to_coco(val, metadata_path, "data/annotations/val.json")
    convert_dmid_to_coco(test, metadata_path, "data/annotations/test.json")


if __name__ == "__main__":
    main()