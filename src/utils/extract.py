import zipfile
from pathlib import Path

def extract_zip(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"[OK] Dataset uncompressed in {output_dir}")