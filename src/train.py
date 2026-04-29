import torch
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.models.fasterrcnn import get_model as get_faster
from src.models.retinanet import get_model as get_retinanet
from src.models.yolo import train_yolo
from src.config import *


def collate_fn(batch):
    return tuple(zip(*batch))


# -------------------------
# FASTERCNN
# -------------------------
def train_fasterrcnn():

    train_dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/train.json",
        "data/raw/TIFF Images",
        max_samples=300
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    model = get_faster(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,   # importante para Faster R-CNN
        momentum=0.9,
        weight_decay=0.0005
    )

    print("\n=== Faster R-CNN TRAIN ===\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:

            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if torch.isnan(loss):
                print("NaN loss skipped")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[Faster R-CNN] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "faster.pth")


# -------------------------
# RETINANET
# -------------------------
def train_retinanet():

    train_dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/train.json",
        "data/raw/TIFF Images",
        max_samples=300
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    model = get_retinanet(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\n=== RetinaNet TRAIN ===\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:

            images = [img.to(DEVICE) for img in images]
            new_targets = []
            for t in targets:
                new_t = {}
                for k, v in t.items():
                    if k == "labels":
                        # convertir 1,2 → 0,1
                        new_t[k] = (v - 1).to(DEVICE)
                    else:
                        new_t[k] = v.to(DEVICE)
                new_targets.append(new_t)

            targets = new_targets

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if torch.isnan(loss):
                print("NaN loss skipped")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[RetinaNet] Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "retina.pth")

# -------------------------
# RT-DETR
# -------------------------
def train_rtdetr():
    from ultralytics import YOLO

    model = YOLO("rtdetr-l.pt")

    model.train(
        data="data.yaml",
        epochs=EPOCHS,
        imgsz=640,
        batch=4
    )

    model.save("rtdetr.pt")


# -------------------------
# MAIN
# -------------------------
def main(model_name):

    if model_name == "yolo":
        train_yolo()

    elif model_name == "faster":
        train_fasterrcnn()

    elif model_name == "retina":
        train_retinanet()

    elif model_name == "rtdetr":
        train_rtdetr()

    else:
        print("Modelo no válido. Usa: yolo | faster | retina  | rtdetr")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()
    main(args.model)
