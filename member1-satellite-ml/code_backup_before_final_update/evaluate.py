import json, torch
from torch.utils.data import DataLoader
from config import *
from src.dataset.oil_dataset import OilSpillDataset
from src.models.unet import UNet
from src.utils.metrics import segmentation_metrics

ds=OilSpillDataset(TEST_IMAGES_DIR,TEST_MASKS_DIR)
loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False)
model=UNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=DEVICE))
model.eval()

allm=[]
with torch.no_grad():
    for b in loader:
        allm.append(segmentation_metrics(model(b["image"].to(DEVICE)), b["mask"].to(DEVICE)))

result={k:sum(m[k] for m in allm)/len(allm) for k in allm[0]}
print(json.dumps(result,indent=2))
(METRICS_OUTPUT_DIR/"model_metrics.json").write_text(json.dumps(result,indent=2))
