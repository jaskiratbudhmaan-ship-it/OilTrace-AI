from pathlib import Path
import cv2, matplotlib.pyplot as plt, numpy as np, sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import *

def count(folder): return len(list(folder.glob("*.png")))

print("Train:",count(TRAIN_IMAGES_DIR),count(TRAIN_MASKS_DIR))
print("Val:",count(VAL_IMAGES_DIR),count(VAL_MASKS_DIR))
print("Test:",count(TEST_IMAGES_DIR),count(TEST_MASKS_DIR))

imgs=sorted(TRAIN_IMAGES_DIR.glob("*.png"))
masks=sorted(TRAIN_MASKS_DIR.glob("*.png"))
if not imgs:
    print("No processed data. Run prepare_dataset.py")
    raise SystemExit(1)

img=cv2.imread(str(imgs[0]),0); m=cv2.imread(str(masks[0]),0)
fig,ax=plt.subplots(1,2,figsize=(8,4))
ax[0].imshow(img,cmap="gray"); ax[0].set_title("SAR image"); ax[0].axis("off")
ax[1].imshow(m,cmap="gray"); ax[1].set_title("Oil mask"); ax[1].axis("off")
fig.tight_layout()
out=DATASET_PREVIEW_DIR/"dataset_preview.png"
fig.savefig(out,dpi=150); plt.close(fig)
print("Preview:",out)
print("✓ PART 2 DATASET PREPARATION SUCCESSFUL")
