import zipfile
from pathlib import Path

def extract_zip(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"[OK] Main zip extracted in {output_dir}")

    extract_nested_zips(output_dir)

def extract_nested_zips(root_dir):
    root_dir = Path(root_dir)

    zip_files = list(root_dir.rglob("*.zip"))

    while zip_files:
        zip_path = zip_files.pop()

        print(f"[INFO] Extracting nested zip: {zip_path}")

        extract_path = zip_path.parent

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except zipfile.BadZipFile:
            print(f"[WARNING] Failed to extract {zip_path}")
            continue

        zip_path.unlink()

        zip_files = list(root_dir.rglob("*.zip"))

    print("[OK] All nested zips extracted")