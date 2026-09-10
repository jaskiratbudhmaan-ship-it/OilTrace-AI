from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images=sorted(Path(images_dir).glob("*.png"))
        self.masks=sorted(Path(masks_dir).glob("*.png"))
        if len(self.images)!=len(self.masks):
            raise ValueError("Image/mask count mismatch")
        if not self.images:
            raise ValueError("No processed images found")
        for a,b in zip(self.images,self.masks):
            if a.name!=b.name: raise ValueError(f"Name mismatch {a.name} {b.name}")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img=cv2.imread(str(self.images[idx]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        mask=cv2.imread(str(self.masks[idx]),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        mask=(mask>0.5).astype(np.float32)
        return {
            "image": torch.from_numpy(img).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "filename": self.images[idx].name
        }
