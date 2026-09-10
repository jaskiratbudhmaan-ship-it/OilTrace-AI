from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import *
def c(p): return len(list(p.glob("*.png")))
print("Extracted images folder:",KAGGLE_IMAGES_DIR.exists())
print("Extracted masks folder :",KAGGLE_MASKS_DIR.exists())
print("Train:",c(TRAIN_IMAGES_DIR),c(TRAIN_MASKS_DIR))
print("Val  :",c(VAL_IMAGES_DIR),c(VAL_MASKS_DIR))
print("Test :",c(TEST_IMAGES_DIR),c(TEST_MASKS_DIR))
print("Device:",DEVICE)
ok=(KAGGLE_IMAGES_DIR.exists() and KAGGLE_MASKS_DIR.exists()
    and c(TRAIN_IMAGES_DIR)>0 and c(VAL_IMAGES_DIR)>0 and c(TEST_IMAGES_DIR)>0)
print("✓ READY FOR TRAINING" if ok else "✗ NOT READY")
