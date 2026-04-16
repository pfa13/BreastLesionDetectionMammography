import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from pathlib import Path
from PIL import Image


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, img_root, max_samples=None):
        self.coco = CocoDetection(img_root, ann_file)
        self.img_root = Path(img_root)

        self.ids = list(range(len(self.coco)))
        if max_samples:
            self.ids = self.ids[:max_samples]

        # 🔥 3 canales correctos (RGB)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        # 🔥 mapping explícito de clases (CRÍTICO PARA RETINANET)
        # COCO original: 1=mass, 2=calcification
        self.class_map = {
            1: 1,
            2: 2
        }

    def __getitem__(self, idx):
        img, target = self.coco[self.ids[idx]]

        file_name = self.coco.coco.imgs[
            self.coco.ids[self.ids[idx]]
        ]["file_name"]

        img_path = self.img_root / file_name
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        boxes = []
        labels = []

        for obj in target:
            x, y, w, h = obj["bbox"]

            # COCO -> XYXY
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = max(x1 + 1, x + w)
            y2 = max(y1 + 1, y + h)

            # filtro básico de coherencia
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            boxes.append([x1, y1, x2, y2])

            # 🔥 FIX CRÍTICO: clases consistentes
            labels.append(self.class_map[obj["category_id"]])

        # 🔥 conversión segura a tensores (OBLIGATORIO)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # seguridad extra
        assert boxes.shape[0] == labels.shape[0]

        # limpieza final
        labels = labels.long()
        boxes = boxes.float()

        return img, {
            "boxes": boxes,
            "labels": labels
        }

    def __len__(self):
        return len(self.ids)