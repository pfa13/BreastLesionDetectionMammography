import torch
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.models.fasterrcnn import get_model as get_faster
from src.models.retinanet import get_model as get_retina
from src.config import *

def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------------
# EVALUATE MODEL (PREDICTIONS)
# -------------------------
def get_predictions(model, loader, device):
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu()
                })
                gts.append({
                    "boxes": tgt["boxes"]
                })

    return preds, gts


# -------------------------
# SIMPLE METRICS
# -------------------------
def compute_metrics(preds, gts, score_thresh=0.3):

    tp, fp, fn = 0, 0, 0

    for p, g in zip(preds, gts):

        keep = p["scores"] > score_thresh
        pboxes = p["boxes"][keep]
        gboxes = g["boxes"]

        if len(pboxes) == 0:
            fn += len(gboxes)
            continue

        if len(gboxes) == 0:
            fp += len(pboxes)
            continue

        tp += min(len(pboxes), len(gboxes))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall


# -------------------------
# YOLO EVAL
# -------------------------
def evaluate_yolo(model_path):
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data="data.yaml")

    return {
        "map50": results.box.map50,
        "map": results.box.map
    }


# -------------------------
# MAIN EVAL
# -------------------------
def run_full_evaluation():

    device = DEVICE

    val_dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/val.json",
        "data/raw/TIFF Images"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("\n=== EVALUACIÓN MODELOS ===\n")

    # ---------------- FASTERCNN ----------------
    from src.models.fasterrcnn import get_model as get_faster

    faster = get_faster(NUM_CLASSES)
    faster.load_state_dict(torch.load("faster.pth", map_location=device))
    faster.to(device)

    preds, gts = get_predictions(faster, val_loader, device)
    p, r = compute_metrics(preds, gts)

    print(f"Faster R-CNN -> Precision: {p:.4f}, Recall: {r:.4f}")

    # ---------------- RETINANET ----------------
    from src.models.retinanet import get_model as get_retina

    retina = get_retina(NUM_CLASSES)
    retina.load_state_dict(torch.load("retina.pth", map_location=device))
    retina.to(device)

    preds, gts = get_predictions(retina, val_loader, device)
    p, r = compute_metrics(preds, gts)

    print(f"RetinaNet -> Precision: {p:.4f}, Recall: {r:.4f}")

    # ---------------- YOLO ----------------
    yolo_score = evaluate_yolo("runs/detect/train/weights/best.pt")

    print(f"YOLO mAP50: {yolo_score['map50']:.4f}")
    print(f"YOLO mAP:   {yolo_score['map']:.4f}")


if __name__ == "__main__":
    run_full_evaluation()