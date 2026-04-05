import os
import json
import pandas as pd

from file_utils import read_image_size


def convert_dmid_to_coco(images, metadata_path, output_json):
    df = pd.read_excel(metadata_path, header=None)

    df.columns = [
        "image_id", "view", "tissue", "abnormality",
        "class", "x", "y", "radius"
    ]

    grouped = df.groupby("image_id")

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
        filename = os.path.splitext(os.path.basename(img_path))[0]

        if filename not in grouped.groups:
            continue

        rows = grouped.get_group(filename)

        size = read_image_size(img_path)
        if size is None:
            continue

        h, w = size

        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": w,
            "height": h
        })

        for _, row in rows.iterrows():

            if pd.isna(row["x"]) or pd.isna(row["y"]) or pd.isna(row["radius"]):
                continue

            try:
                x = float(row["x"])
                y = float(row["y"])
                r = float(row["radius"])
            except:
                continue

            # bounding box
            x_min = x - r
            y_min = y - r
            width = 2 * r
            height = 2 * r

            # category
            abnormality = str(row["abnormality"]).upper()

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

    print(f"[OK] COCO guardado en {output_json}")