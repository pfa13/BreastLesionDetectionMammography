import os
import json
import torch
import numpy as np
import pydicom
from PIL import Image
from src.utils.file_utils import load_image

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.annotation_file = annotation_file

        self.ann_by_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.ann_by_image.setdefault(img_id, []).append(ann)

        self.images = [
            img for img in self.images
            if img["id"] in self.ann_by_image and len(self.ann_by_image[img["id"]]) > 0
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = img_info["file_name"]

        img = load_image(img_path)

        anns = self.ann_by_image.get(img_info["id"], [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target