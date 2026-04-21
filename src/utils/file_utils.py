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
            images.append(str(p.resolve()))

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


# 🔥 SAFE NORMALIZER (CRÍTICO)
def safe_normalize(img):
    img = img.astype(np.float32)

    if np.isnan(img).any():
        return None

    if np.isinf(img).any():
        return None

    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    denom = img.max() - img.min()
    if denom < 1e-6:
        return None

    img = (img - img.min()) / (denom + 1e-8)
    img = (img * 255).astype(np.uint8)

    return img


def read_image(img_path):
    ext = Path(img_path).suffix.lower()

    # ---------------- DICOM ----------------
    if ext == ".dcm":
        try:
            dcm = pydicom.dcmread(img_path)
            img = dcm.pixel_array.astype(np.float32)

            if img.ndim > 2:
                img = img.squeeze()

            img = safe_normalize(img)
            if img is None:
                return None

            return np.stack([img] * 3, axis=-1)

        except:
            return None

    # ---------------- TIFF ----------------
    if ext in [".tif", ".tiff"]:
        try:
            img = tifffile.imread(img_path)

            if img.ndim > 2:
                img = img.squeeze()

            img = img.astype(np.float32)

            img = safe_normalize(img)
            if img is None:
                return None

            return np.stack([img] * 3, axis=-1)

        except:
            return None

    # ---------------- RGB / GRAY ----------------
    try:
        img = Image.open(img_path).convert("L")
        img = np.array(img)

        img = safe_normalize(img)
        if img is None:
            return None

        return np.stack([img] * 3, axis=-1)

    except:
        return None