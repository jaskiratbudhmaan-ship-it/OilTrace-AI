from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self,images_dir,masks_dir):
        self.images=sorted(Path(images_dir).glob("*.png"))
        self.masks=sorted(Path(masks_dir).glob("*.png"))
        if len(self.images)!=len(self.masks) or not self.images:
            raise ValueError("Processed image/mask dataset missing or mismatched")
        for a,b in zip(self.images,self.masks):
            if a.name!=b.name: raise ValueError(f"Name mismatch: {a.name} vs {b.name}")
    def __len__(self): return len(self.images)
    def __getitem__(self,i):
        x=cv2.imread(str(self.images[i]),0).astype(np.float32)/255.0
        y=cv2.imread(str(self.masks[i]),0).astype(np.float32)/255.0
        y=(y>0.5).astype(np.float32)
        return {"image":torch.from_numpy(x).unsqueeze(0),
                "mask":torch.from_numpy(y).unsqueeze(0),
                "filename":self.images[i].name}
