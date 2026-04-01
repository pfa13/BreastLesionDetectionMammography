import os
import json
import pandas as pd
from PIL import Image
from file_utils import read_image

def convert_dmid_to_coco(images, metadata_path, output_json):
    df = pd.read_excel(metadata_path, header=None)

    df.columns = [
        "image_id", "view", "tissue", "abnormality",
        "class", "x", "y", "radius"
    ]

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "mass", "supercategory": "lesion"},
            {"id": 2, "name": "calcification", "supercategory": "lesion"}
        ]
    }

    ann_id = 0
    img_id = 0

    for img_path in images:
        filename = os.path.splitext(os.path.basename(img_path))[0]

        rows = df[df["image_id"] == filename]

        if len(rows) == 0:
            continue

        img = read_image(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        coco["images"].append({
            "id": img_id,
            "file_name": img_path,
            "width": w,
            "height": h
        })

        for _, row in rows.iterrows():
            if row["abnormality"] == "NORM":
                continue

            try:
                x = float(row["x"])
                y = float(row["y"])
                r = float(row["radius"])
            except (ValueError, TypeError):
                continue

            # Bounding box
            x_min = max(0, x - r)
            y_min = max(0, y - r)
            width = min(2 * r, w - x_min)
            height = min(2 * r, h - y_min)

            # Clasificación simple
            abnormality = str(row["abnormality"])

            if "CALC" in abnormality:
                category_id = 2
            else:
                category_id = 1

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