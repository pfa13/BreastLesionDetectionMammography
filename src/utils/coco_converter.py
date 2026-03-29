import os
import json
import pandas as pd
import cv2

def convert_dmid_to_coco(images, metadata_path, output_json):
    df = pd.read_excel(metadata_path)  # o read_csv si es csv

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "mass"},
            {"id": 2, "name": "calcification"}
        ]
    }

    ann_id = 0
    img_id = 0

    for img_path in images:
        filename = os.path.basename(img_path)

        # buscar metadata de esa imagen
        rows = df[df["image_name"] == filename]

        if len(rows) == 0:
            continue

        # cargar imagen para tamaño
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        for _, row in rows.iterrows():
            x = row["center_x"]
            y = row["center_y"]
            r = row["radius"]

            x_min = x - r
            y_min = y - r
            width = 2 * r
            height = 2 * r

            # categoría (adaptar según dataset)
            label = row.get("label", "mass")

            category_id = 1 if label == "mass" else 2

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })

            ann_id += 1

        img_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"[OK] COCO saved in {output_json}")