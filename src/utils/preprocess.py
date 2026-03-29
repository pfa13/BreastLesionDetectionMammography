import argparse
from pathlib import Path

from utils.extract import extract_zip
from utils.split import create_splits, save_splits
from utils.coco_converter import convert_to_coco
from utils.file_utils import collect_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al .zip")
    args = parser.parse_args()

    raw_dir = "data/raw"
    splits_dir = "data/splits"
    annotations_dir = "data/annotations"

    # 1. Extraer
    extract_zip(args.input, raw_dir)

    # 2. Buscar imágenes
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
    Path(annotations_dir).mkdir(parents=True, exist_ok=True)

    convert_to_coco(train, f"{annotations_dir}/train.json")
    convert_to_coco(val, f"{annotations_dir}/val.json")
    convert_to_coco(test, f"{annotations_dir}/test.json")


if __name__ == "__main__":
    main()