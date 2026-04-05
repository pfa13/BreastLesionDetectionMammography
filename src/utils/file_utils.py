from pathlib import Path
import cv2
from PIL import Image
import pydicom
import numpy as np

def read_image_size(img_path):
    ext = img_path.lower().split('.')[-1]

    # --- DICOM (MUY IMPORTANTE) ---
    if ext == "dcm":
        try:
            dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
            h = int(dicom.Rows)
            w = int(dicom.Columns)
            return h, w
        except Exception as e:
            print(f"[WARNING] Error leyendo tamaño DICOM {img_path}: {e}")
            return None

    # --- IMÁGENES normales (rápido con PIL) ---
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            return h, w
    except Exception as e:
        print(f"[WARNING] No se pudo leer tamaño de {img_path}: {e}")
        return None

def read_image(img_path):
    ext = img_path.lower().split('.')[-1]

    # --- DICOM ---
    if ext == "dcm":
        try:
            dicom = pydicom.dcmread(img_path)
            img = dicom.pixel_array.astype(np.float32)

            # Normalization
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to 8 bits
            img = (img * 255).astype(np.uint8)

            # Convert to 3 chanels
            img = np.stack([img]*3, axis=-1)

            return img

        except Exception as e:
            print(f"[WARNING] Error leyendo DICOM {img_path}: {e}")
            return None

    # --- OpenCV ---
    img = cv2.imread(img_path)
    if img is not None:
        return img

    # --- PIL fallback ---
    try:
        img = Image.open(img_path).convert("RGB")
        return np.array(img)
    except:
        print(f"[WARNING] No se pudo leer {img_path}")
        return None

def collect_images(root_dir, extensions={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}):
    root_dir = Path(root_dir)

    images = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in extensions:
            continue

        path_str = str(path).lower()

        # excluir carpetas no válidas
        if "roi masks" in path_str:
            continue
        if "mask" in path_str:
            continue
        if "annotation" in path_str:
            continue
        if "pixel-level" in path_str:
            continue

        images.append(str(path))

    return images