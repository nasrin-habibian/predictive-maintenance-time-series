"""Download and extract the NASA C-MAPSS turbofan benchmark."""
from pathlib import Path
import zipfile
import requests
from .config import RAW_DIR

# NASA Prognostics Center of Excellence public C-MAPSS archive.
DATA_URL = (
    "https://ti.arc.nasa.gov/c/6/"
    "C-MAPSSData.zip"
)

def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    required = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    if all((RAW_DIR / name).exists() for name in required):
        print("FD001 files already exist; download skipped.")
        return

    archive = RAW_DIR / "C-MAPSSData.zip"
    print(f"Downloading dataset from {DATA_URL}")
    response = requests.get(DATA_URL, timeout=90)
    response.raise_for_status()
    archive.write_bytes(response.content)

    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            "The downloaded file is not a ZIP archive. "
            "Download C-MAPSS manually from NASA and place the FD001 files in data/raw/."
        )

    with zipfile.ZipFile(archive) as zf:
        members = {Path(name).name: name for name in zf.namelist()}
        missing = [name for name in required if name not in members]
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {missing}")
        for name in required:
            target = RAW_DIR / name
            target.write_bytes(zf.read(members[name]))

    print("Dataset ready:", ", ".join(required))

if __name__ == "__main__":
    download_dataset()
