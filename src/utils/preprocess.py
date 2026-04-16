import argparse
from pathlib import Path

from src.utils.extract import extract_zip
from src.utils.file_utils import collect_images, read_image_size
from src.utils.coco_converter import convert_dmid_to_coco
from src.utils.coco_kfold import create_coco_kfold, save_coco_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al .zip")
    args = parser.parse_args()

    raw_dir = Path("./data/raw")
    annotations_dir = Path("./data/annotations")
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extraer dataset
    extract_zip(args.input, raw_dir)

    # 2. Usar SOLO TIFF Images
    tiff_dir = raw_dir / "TIFF Images"
    images = collect_images(tiff_dir)

    print(f"[INFO] Imágenes encontradas: {len(images)}")

    # 3. Filtrar imágenes inválidas
    valid_images = []
    for img in images:
        if read_image_size(img) is not None:
            valid_images.append(img)
        else:
            print(f"[WARNING] Imagen inválida: {img}")

    images = valid_images
    print(f"[INFO] Imágenes válidas: {len(images)}")

    # 4. Convertir a COCO
    metadata_path = raw_dir / "Metadata.xlsx"
    full_coco_path = annotations_dir / "full.json"

    convert_dmid_to_coco(images, metadata_path, full_coco_path)

    # 5. Crear K-Fold
    folds = create_coco_kfold(full_coco_path, k=10)

    # 6. Guardar folds
    save_coco_folds(folds, annotations_dir)

    print("[OK] Pipeline finished")


if __name__ == "__main__":
    main()