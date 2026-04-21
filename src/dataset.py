import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.utils.file_utils import read_image


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json, img_root, max_samples=None):
        import json

        with open(coco_json) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.img_root = Path(img_root)

        self.img_to_anns = {}
        for ann in self.annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.images = self.images[:max_samples] if max_samples else self.images

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):

        img_info = self.images[idx]
        img_path = img_info["file_name"]

        image = read_image(img_path)
        if image is None:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        else:
            image = self.transform(image)

        anns = self.img_to_anns.get(img_info["id"], [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # sanity check
            if w <= 1 or h <= 1:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = x + w
            y2 = y + h

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

    def __len__(self):
        return len(self.images)