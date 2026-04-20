import torch
from torch.utils.data import DataLoader
import numpy as np
from ultralytics import YOLO

from src.dataset import CocoDataset
from src.config import *

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, loader, device):
    model.eval()

    losses = []

    with torch.no_grad():
        for images, targets in loader:

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images, targets)

            loss = sum(loss for loss in outputs.values())
            losses.append(loss.item())

    return np.mean(losses)

def evaluate_yolo(model_path):
    model = YOLO(model_path)

    results = model.val(data="data.yaml")

    return {
        "map50": results.box.map50,
        "map": results.box.map
    }

def run_full_evaluation():
    device = DEVICE

    train_dataset = CocoDataset(
        ann_file=f"{ANNOTATIONS_DIR}/train.json",
        img_root="data/raw/TIFF Images"
    )

    val_dataset = CocoDataset(
        ann_file=f"{ANNOTATIONS_DIR}/val.json",
        img_root="data/raw/TIFF Images"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("\n=== EVALUACIÓN MODELOS ===\n")

    # Faster R-CNN
    from src.models.fasterrcnn import get_model as get_faster

    faster = get_faster(NUM_CLASSES)
    faster.load_state_dict(torch.load("faster.pth", map_location=device))
    faster.to(device)

    faster_score = evaluate_model(faster, val_loader, device)

    print(f"Faster R-CNN Loss: {faster_score:.4f}")

    # RetinaNet
    from src.models.retinanet import get_model as get_retina

    retina = get_retina(NUM_CLASSES)
    retina.load_state_dict(torch.load("retina.pth", map_location=device))
    retina.to(device)

    retina_score = evaluate_model(retina, val_loader, device)

    print(f"RetinaNet Loss: {retina_score:.4f}")

    # YOLO
    yolo_score = evaluate_yolo("runs/detect/train/weights/best.pt")

    print(f"YOLO mAP50: {yolo_score['map50']:.4f}")
    print(f"YOLO mAP:   {yolo_score['map']:.4f}")