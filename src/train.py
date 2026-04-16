import torch
from torch.utils.data import DataLoader

from src.dataset import CocoDataset
from src.models.fasterrcnn import get_model as get_faster
from src.models.yolo import train_yolo
from src.models.retinanet import get_model as get_retinanet
from src.config import *

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            total_loss += loss.item()

    return total_loss / len(loader)

def train_fasterrcnn():
    train_dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/train.json",
        "data/raw/TIFF Images",
        max_samples=300 
    )

    val_dataset = CocoDataset(
        f"{ANNOTATIONS_DIR}/val.json",
        "data/raw/TIFF Images",
        max_samples=100
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,   # CPU FIX
        collate_fn=collate_fn
    )

    model = get_faster(NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )

    print("Starting epochs")
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for images, targets in train_loader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"[FAST FasterRCNN] Epoch {epoch} Loss: {total_loss:.4f}")

def train_retinanet():
    train_dataset = CocoDataset(
        ann_file=f"{ANNOTATIONS_DIR}/train.json",
        img_root="data/raw/TIFF Images"
    )

    val_dataset = CocoDataset(
        ann_file=f"{ANNOTATIONS_DIR}/val.json",
        img_root="data/raw/TIFF Images"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_retinanet()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"[RetinaNet] Epoch {epoch}")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}")


def main(model_name):
    if model_name == "yolo":
        train_yolo()

    elif model_name == "faster":
        train_fasterrcnn()

    elif model_name == "retina":
        train_retinanet()

    else:
        print("Modelo no válido")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    main(args.model)