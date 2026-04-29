import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

from src.utils.file_utils import read_image_size


def is_valid_box(x, y, w, h, img_w, img_h):
    if any(map(lambda v: v is None, [x, y, w, h])):
        return False

    if any(map(lambda v: str(v) == "nan", [x, y, w, h])):
        return False

    if w <= 1 or h <= 1:
        return False

    if w > img_w * 1.5 or h > img_h * 1.5:
        return False

    if x < 0 or y < 0:
        return False

    if x > img_w or y > img_h:
        return False

    return True


def convert_dmid_to_coco(images, metadata_path, output_json):

    df = pd.read_excel(metadata_path, header=None)
    df = df.dropna()  # ELIMINA NAN DIRECTOS

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "mass"},
            {"id": 2, "name": "calcification"}
        ]
    }

    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[str(row[0])].append(row)

    image_map = {Path(img).stem: img for img in images}

    ann_id = 0
    img_id = 0

    for filename, rows in grouped.items():

        if filename not in image_map:
            continue

        img_path = image_map[filename]

        size = read_image_size(img_path)
        if size is None:
            continue

        img_h, img_w = size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path,
            "width": img_w,
            "height": img_h
        })

        for row in rows:
            try:
                x = float(row.iloc[5])
                y = float(row.iloc[6])
                r = float(row.iloc[7])
            except:
                continue

            if r <= 0 or str(r) == "nan":
                continue

            x_min = x - r
            y_min = y - r
            w_box = 2 * r
            h_box = 2 * r

            # clamp a imagen
            x_min = max(0, x_min)
            y_min = max(0, y_min)

            w_box = min(w_box, img_w - x_min)
            h_box = min(h_box, img_h - y_min)

            if not is_valid_box(x_min, y_min, w_box, h_box, img_w, img_h):
                continue

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

    print(f"[OK] Clean COCO saved: {output_json}")