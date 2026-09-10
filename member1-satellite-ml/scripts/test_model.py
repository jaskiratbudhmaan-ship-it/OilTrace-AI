from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from config import DEVICE,UNET_BASE_CHANNELS
from src.models.unet import UNet
m=UNet(base_channels=UNET_BASE_CHANNELS).to(DEVICE)
x=torch.randn(2,1,256,256).to(DEVICE)
with torch.no_grad(): y=m(x)
print("Input:",x.shape,"Output:",y.shape,"Device:",DEVICE)
assert tuple(y.shape)==(2,1,256,256)
print("✓ U-NET MODEL WORKING")
