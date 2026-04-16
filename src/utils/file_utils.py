from pathlib import Path
from PIL import Image
import numpy as np
import pydicom
import tifffile

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}


def collect_images(root_dir):
    root_dir = Path(root_dir)
    images = []

    for p in root_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            images.append(str(p.resolve()))  # 🔥 ruta absoluta

    return images


def read_image_size(img_path):
    ext = Path(img_path).suffix.lower()

    if ext == ".dcm":
        try:
            dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
            return int(dicom.Rows), int(dicom.Columns)
        except:
            return None

    try:
        with Image.open(img_path) as img:
            return img.height, img.width
    except:
        return None


def read_image(img_path):
    ext = Path(img_path).suffix.lower()

    if ext == ".dcm":
        try:
            dcm = pydicom.dcmread(img_path)
            img = dcm.pixel_array.astype(np.float32)

            if img.ndim > 2:
                img = img.squeeze()

            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip(img, p1, p99)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)

            return np.stack([img]*3, axis=-1)

        except Exception as e:
            print(f"[WARNING] DICOM error {img_path}: {e}")
            return None

    if ext in [".tif", ".tiff"]:
        try:
            img = tifffile.imread(img_path)

            if img.ndim > 2:
                img = img.squeeze()

            img = img.astype(np.float32)

            p1, p99 = np.percentile(img, (1, 99))
            img = np.clip(img, p1, p99)

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)

            return np.stack([img]*3, axis=-1)

        except Exception as e:
            print(f"[WARNING] TIFF error {img_path}: {e}")
            return None

    try:
        img = Image.open(img_path).convert("L")
        img = np.array(img)
        return np.stack([img]*3, axis=-1)
    except Exception as e:
        print(f"[WARNING] Error leyendo {img_path}: {e}")
        return None