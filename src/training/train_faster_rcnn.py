import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.datasets.coco_dataset import COCODataset
from src.models.faster_rcnn import get_model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = "./data/raw/TIFF Images"
    train_ann = "./data/annotations/fold_0/train.json"
    val_ann = "./data/annotations/fold_0/val.json"

    transforms = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ])

    train_dataset = COCODataset(images_dir, train_ann, transforms)
    val_dataset = COCODataset(images_dir, val_ann, transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = get_model(num_classes=3)  # 2 clases + background
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss:.4f}")

        if epoch % 2 == 0:
            evaluate(model, val_loader, device, fold_idx=0, epoch=epoch)

    torch.save(model.state_dict(), "faster_rcnn.pth")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(model, data_loader, device, fold_idx, epoch):
    model.eval()

    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            for target, output in zip(targets, outputs):
                img_id = int(target["image_id"].item())

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box

                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2 - x1),
                            float(y2 - y1)
                        ],
                        "score": float(score)
                    })

    import json
    pred_file = f"predictions_fold{fold_idx}.json"

    with open(pred_file, "w") as f:
        json.dump(results, f)

    coco_gt = COCO(data_loader.dataset.annotation_file)
    if len(results) == 0:
        print("[WARNING] No detections, skipping evaluation")
        return
    
    coco_dt = coco_gt.loadRes(pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP@[0.5:0.95]": float(coco_eval.stats[0]),
        "mAP@0.5": float(coco_eval.stats[1]),
        "mAP_small": float(coco_eval.stats[3]),
        "mAP_medium": float(coco_eval.stats[4]),
        "mAP_large": float(coco_eval.stats[5]),
    }

    metrics_file = f"metrics_fold{fold_idx}_epoch{epoch}.json"

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[METRICS] {metrics}")

if __name__ == "__main__":
    main()