from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from config import *
from src.dataset.oil_dataset import OilSpillDataset

ds=OilSpillDataset(TRAIN_IMAGES_DIR,TRAIN_MASKS_DIR)
batch=next(iter(DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)))
print("Image batch:",batch["image"].shape)
print("Mask batch:",batch["mask"].shape)
print("✓ PYTORCH DATASET LOADER WORKING")
