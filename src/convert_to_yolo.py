import json
from pathlib import Path
import shutil
from PIL import Image

def find_image(img_name):
    raw_dir = Path("data/raw")

    img_name = Path(img_name).name

    # buscar por nombre exacto
    for p in raw_dir.rglob("*"):
        if p.name == img_name:
            return p

    name_no_ext = Path(img_name).stem

    for p in raw_dir.rglob("*"):
        if p.stem == name_no_ext:
            return p

    return None

def convert_coco_to_yolo(coco_json, images_dir, output_dir):
    with open(coco_json) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img_info = images[img_id]

        img_name = img_info["file_name"]

        img_path = find_image(img_name)

        if img_path is None:
            print(f"[WARNING] Imagen no encontrada: {img_name}")
            continue

        # copiar imagen
        dst = output_dir / "images" / img_path.name

        try:
            img = Image.open(img_path).convert("RGB")  # 🔥 CLAVE
            img.save(dst)
        except Exception as e:
            print(f"[ERROR] No se pudo procesar {img_path}: {e}")
            continue

        try:
            img = Image.open(img_path)
            width, height = img.size
        except:
            print(f"[WARNING] No se pudo abrir {img_name}")
            continue

        # copiar imagen
        dst = output_dir / "images" / img_name
        if not dst.exists():
            shutil.copy(img_path, dst)

        x, y, w, h = ann["bbox"]

        # evitar basura
        if w <= 1 or h <= 1:
            continue

        # COCO → YOLO
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        class_id = ann["category_id"] - 1

        label_path = output_dir / "labels" / f"{Path(img_name).stem}.txt"

        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")


def main():
    fold = "fold_0"  # 🔥 cambia aquí si quieres otro

    base_ann = Path(f"data/annotations/{fold}")

    convert_coco_to_yolo(
        base_ann / "train.json",
        "data/raw/TIFF Images",
        f"data/yolo/{fold}/train"
    )

    convert_coco_to_yolo(
        base_ann / "val.json",
        "data/raw/TIFF Images",
        f"data/yolo/{fold}/val"
    )


if __name__ == "__main__":
    main()