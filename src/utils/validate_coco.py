import json
import numpy as np


def validate_coco(path):

    print(f"\n🔍 Validating COCO: {path}\n")

    with open(path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}

    errors = 0
    warnings = 0

    for ann in data["annotations"]:

        img_id = ann["image_id"]

        if img_id not in images:
            print(f"❌ Missing image_id: {img_id}")
            errors += 1
            continue

        bbox = ann.get("bbox", None)

        if bbox is None:
            print(f"⚠️ Missing bbox in ann {ann['id']}")
            warnings += 1
            continue

        x, y, w, h = bbox

        # 🔥 invalid geometry
        if w <= 0 or h <= 0:
            print(f"❌ Invalid bbox (neg/zero): {bbox}")
            errors += 1

        if np.isnan([x, y, w, h]).any():
            print(f"❌ NaN in bbox: {bbox}")
            errors += 1

        if np.isinf([x, y, w, h]).any():
            print(f"❌ Inf in bbox: {bbox}")
            errors += 1

        # optional sanity check
        if w > 1000 or h > 1000:
            print(f"⚠️ Suspicious large bbox: {bbox}")
            warnings += 1

    print("\n=== SUMMARY ===")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    if errors > 0:
        print("❌ COCO dataset is NOT safe for training")
    else:
        print("✅ COCO dataset looks OK")


if __name__ == "__main__":
    validate_coco("data/annotations/fold_0/train.json")