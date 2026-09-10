from pathlib import Path
import sys, requests
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DOWNLOADS_DIR, ZENODO_DOWNLOAD_URL, ZENODO_ARCHIVE_NAME

def main():
    target = DOWNLOADS_DIR / ZENODO_ARCHIVE_NAME
    if target.exists() and target.stat().st_size > 100_000_000:
        print("Dataset already exists:", target)
        return

    print("Downloading real Sentinel-1 oil spill dataset from Zenodo...")
    with requests.get(ZENODO_DOWNLOAD_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(target, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print("Saved to:", target)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Automatic download failed:", e)
        print("Download manually from https://zenodo.org/records/4672426")
        print("Place Radar_data.rar inside data/downloads/")
