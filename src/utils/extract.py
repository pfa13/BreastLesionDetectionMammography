import zipfile
from pathlib import Path

def extract_zip(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    seen = set()

    while True:
        zips = [z for z in output_dir.rglob("*.zip") if z not in seen]

        if not zips:
            break

        for z in zips:
            print(f"[INFO] Extracting nested zip: {z}")
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(z.parent)
            seen.add(z)
            z.unlink()

    print("[OK] All nested zips extracted")