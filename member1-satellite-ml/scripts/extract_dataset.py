from pathlib import Path
import shutil, subprocess, sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DOWNLOADS_DIR, EXTRACTED_DIR, ZENODO_ARCHIVE_NAME

archive = DOWNLOADS_DIR / ZENODO_ARCHIVE_NAME

if not archive.exists():
    print("Missing:", archive)
    raise SystemExit(1)

if shutil.which("unar"):
    subprocess.run(["unar","-force-overwrite","-output-directory",str(EXTRACTED_DIR),str(archive)], check=True)
elif shutil.which("unrar"):
    subprocess.run(["unrar","x","-o+",str(archive),str(EXTRACTED_DIR)+"/"], check=True)
elif shutil.which("bsdtar"):
    subprocess.run(["bsdtar","-xf",str(archive),"-C",str(EXTRACTED_DIR)], check=True)
else:
    print("No RAR extractor found.")
    print("Install on Mac with: brew install unar")
    raise SystemExit(1)

print("Extracted to:", EXTRACTED_DIR)
