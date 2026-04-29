import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import CocoDataset
from src.config import *
from src.visualize import show_predictions


def collate_fn(batch):
    return tuple(zip(*batch))


# =====================================================
# CONFUSION MATRIX PLOT
# =====================================================
def plot_confusion_matrix(cm, title="Classification Confusion Matrix"):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)

    classes = ["mass", "calc"]
    ticks = np.arange(len(classes))

    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.show()


# =====================================================
# IOU
# =====================================================
def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# =====================================================
# PREDICTIONS
# =====================================================
def get_predictions(model, loader, device):
    model.eval()
    preds, gts = [], []

    with torch.no_grad():
        for images, targets in loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):

                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu()
                })

                gts.append({
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu()
                })
    total_preds = sum(len(p["boxes"]) for p in preds)
    print(f"Total predictions: {total_preds}")
    return preds, gts


# =====================================================
# DETECTION METRICS
# =====================================================
def compute_detection_metrics(preds, gts, score_thresh=0.01, iou_thresh=0.3):

    tp, fp, fn = 0, 0, 0

    for p, g in zip(preds, gts):

        keep = p["scores"] > score_thresh
        pboxes = p["boxes"][keep].tolist()

        gboxes = g["boxes"].tolist()
        matched = set()

        for pb in pboxes:

            best_iou = 0
            best_j = -1

            for j, gb in enumerate(gboxes):
                if j in matched:
                    continue

                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_j = j

            if best_iou >= iou_thresh:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1

        fn += len(gboxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall


# =====================================================
# CLASSIFICATION METRICS
# =====================================================
def compute_classification_metrics(preds, gts, score_thresh=0.3, iou_thresh=0.5):

    tp, fp, fn = 0, 0, 0
    cm = [[0, 0], [0, 0]]

    for p, g in zip(preds, gts):

        keep = p["scores"] > score_thresh

        pboxes = p["boxes"][keep].tolist()
        plabels = p["labels"][keep].tolist()

        gboxes = g["boxes"].tolist()
        glabels = g["labels"].tolist()

        matched_gt = set()

        for pi, pb in enumerate(pboxes):

            best_iou = 0
            best_j = -1

            for j, gb in enumerate(gboxes):
                if j in matched_gt:
                    continue

                current_iou = iou(pb, gb)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_j = j

            if best_iou >= iou_thresh:

                pred_class = plabels[pi]
                gt_class = glabels[best_j]

                # si tu dataset usa 1..N, pero modelo usa 0..N-1
                pred_class = int(pred_class) + 1
                gt_class = int(gt_class)

                cm[gt_class - 1][pred_class - 1] += 1

                if pred_class == gt_class:
                    tp += 1
                else:
                    fp += 1

                matched_gt.add(best_j)

            else:
                fp += 1

        fn += len(gboxes) - len(matched_gt)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall, cm


# =====================================================
# YOLO
# =====================================================
def evaluate_yolo(model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model.val(data="data.yaml")

    return {
        "map50": results.box.map50,
        "map": results.box.map
    }

# =====================================================
# RT DETR
# =====================================================
def evaluate_rtdetr(model_path):
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data="data.yaml")

    return {
        "map50": results.box.map50,
        "map": results.box.map
    }


# =====================================================
# MAIN
# =====================================================
def run_full_evaluation():

    device = DEVICE

    dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/val.json",
        "data/raw/TIFF Images"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("\n=== EVALUACIÓN MODELOS ===\n")

    # =====================================================
    # FASTERR-CNN
    # =====================================================
    """
    from src.models.fasterrcnn import get_model as get_faster

    model = get_faster(NUM_CLASSES)
    model.load_state_dict(torch.load("faster.pth", map_location=device))
    model.to(device)

    preds, gts = get_predictions(model, loader, device)

    det_p, det_r = compute_detection_metrics(preds, gts)
    cls_p, cls_r, cm = compute_classification_metrics(preds, gts)

    print("Faster R-CNN")
    print(f"Detection     -> P: {det_p:.4f} | R: {det_r:.4f}")
    print(f"Classification -> P: {cls_p:.4f} | R: {cls_r:.4f}")

    plot_confusion_matrix(cm)

    show_predictions(model, dataset, device, num_images=3)
    """

    # =====================================================
    # RETINANET
    # =====================================================
    """
    from src.models.retinanet import get_model as get_retina

    model = get_retina(NUM_CLASSES)
    model.load_state_dict(torch.load("retina.pth", map_location=device))
    model.to(device)

    preds, gts = get_predictions(model, loader, device)

    det_p, det_r = compute_detection_metrics(preds, gts)
    cls_p, cls_r, cm = compute_classification_metrics(preds, gts)

    print("RetinaNet")
    print(f"Detection     -> P: {det_p:.4f} | R: {det_r:.4f}")
    print(f"Classification -> P: {cls_p:.4f} | R: {cls_r:.4f}")

    plot_confusion_matrix(cm)

    show_predictions(model, dataset, device, num_images=3)
    """


    # =====================================================
    # YOLO
    # =====================================================
    """
    yolo_score = evaluate_yolo("yolo26n.pt")

    print("YOLO")
    print(f"mAP50: {yolo_score['map50']:.4f}")
    print(f"mAP:   {yolo_score['map']:.4f}")
    """
    # =====================================================
    # RT DETR
    # =====================================================
    
    rtdetr_score = evaluate_rtdetr("rtdetr.pt")

    print("RT-DETR")
    print(f"mAP50: {rtdetr_score['map50']:.4f}")
    print(f"mAP:   {rtdetr_score['map']:.4f}")
    


if __name__ == "__main__":
    run_full_evaluation()
