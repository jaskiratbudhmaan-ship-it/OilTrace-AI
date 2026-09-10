import json, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from src.dataset.oil_dataset import OilSpillDataset
from src.models.unet import UNet
from src.utils.metrics import batch_confusion, metrics_from_counts

ds=OilSpillDataset(TEST_IMAGES_DIR,TEST_MASKS_DIR)
loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
model=UNet(base_channels=UNET_BASE_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=DEVICE)); model.eval()
tp=fp=fn=tn=0
with torch.no_grad():
    for b in tqdm(loader,desc="Test evaluation"):
        z=model(b["image"].to(DEVICE)); y=b["mask"].to(DEVICE)
        a,b1,c,d=batch_confusion(z,y,MASK_THRESHOLD)
        tp+=a; fp+=b1; fn+=c; tn+=d
m=metrics_from_counts(tp,fp,fn,tn); m["test_samples"]=len(ds)
METRICS_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
(METRICS_OUTPUT_DIR/"model_metrics.json").write_text(json.dumps(m,indent=2))
print(json.dumps(m,indent=2))
