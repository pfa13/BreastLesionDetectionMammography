import argparse
from pathlib import Path

from src.utils.extract import extract_zip
from src.utils.file_utils import collect_images
from src.utils.coco_converter import convert_dmid_to_coco
from src.utils.coco_kfold import create_coco_kfold, save_coco_folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al .zip")
    args = parser.parse_args()

    # Rutas
    raw_dir = Path("./data/raw")
    annotations_dir = Path("./data/annotations")

    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Extract dataset (include nested zips)
    extract_zip(args.input, raw_dir)

    # Collect valid images (TIFF + DICOM, without masks)
    images = collect_images(raw_dir)
    print(f"[INFO] {len(images)} imágenes válidas encontradas")

    # Convert a COCO
    metadata_path = raw_dir / "Metadata.xlsx"

    full_coco_path = annotations_dir / "full.json"

    convert_dmid_to_coco(
        images,
        metadata_path,
        full_coco_path
    )

    print("[OK] COCO generated")

    # Create K-Fold
    folds = create_coco_kfold(full_coco_path, k=10)

    # Save folds
    save_coco_folds(folds, annotations_dir)

    print("[OK] Pipeline finished")


if __name__ == "__main__":
    main()