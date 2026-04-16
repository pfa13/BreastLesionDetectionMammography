import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

from src.utils.file_utils import read_image_size


def convert_dmid_to_coco(images, metadata_path, output_json):
    df = pd.read_excel(metadata_path, header=None)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "mass"},
            {"id": 2, "name": "calcification"}
        ]
    }

    # Agrupar metadata por nombre de imagen
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[str(row[0])].append(row)

    # Crear mapping filename → ruta real
    image_map = {
        Path(img).stem: img
        for img in images
    }

    ann_id = 0
    img_id = 0

    for filename, rows in grouped.items():

        if filename not in image_map:
            print(f"[WARNING] No encontrada en TIFF: {filename}")
            continue

        img_path = image_map[filename]

        if not Path(img_path).exists():
            print(f"[WARNING] Ruta inválida: {img_path}")
            continue

        size = read_image_size(img_path)
        if size is None:
            print(f"[WARNING] Imagen inválida: {img_path}")
            continue

        h, w = size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path,
            "width": w,
            "height": h
        })

        for row in rows:
            try:
                x = float(row.iloc[5])
                y = float(row.iloc[6])
                r = float(row.iloc[7])
            except:
                continue

            if r <= 0:
                continue

            x_min = max(0, x - r)
            y_min = max(0, y - r)

            w_box = min(2 * r, w - x_min)
            h_box = min(2 * r, h - y_min)

            label = str(row.iloc[3]).upper()
            category_id = 2 if "CALC" in label else 1

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, w_box, h_box],
                "area": float(w_box * h_box),
                "iscrowd": 0
            })

            ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[OK] COCO saved: {output_json}")