import json
import torch
from torch.utils.data import DataLoader
from config import *
from src.dataset.oil_dataset import OilSpillDataset
from src.models.unet import UNet
from src.utils.losses import combined_loss
from src.utils.metrics import segmentation_metrics

train_ds=OilSpillDataset(TRAIN_IMAGES_DIR,TRAIN_MASKS_DIR)
val_ds=OilSpillDataset(VAL_IMAGES_DIR,VAL_MASKS_DIR)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

model=UNet().to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

best=-1
history=[]

for epoch in range(1,NUM_EPOCHS+1):
    model.train(); total=0
    for b in train_loader:
        x=b["image"].to(DEVICE); y=b["mask"].to(DEVICE)
        opt.zero_grad()
        logits=model(x)
        loss=combined_loss(logits,y)
        loss.backward(); opt.step()
        total+=loss.item()

    model.eval(); vals=[]
    with torch.no_grad():
        for b in val_loader:
            x=b["image"].to(DEVICE); y=b["mask"].to(DEVICE)
            vals.append(segmentation_metrics(model(x),y)["dice"])
    vdice=sum(vals)/len(vals) if vals else 0
    avg=total/max(1,len(train_loader))
    print(f"Epoch {epoch}/{NUM_EPOCHS} loss={avg:.4f} val_dice={vdice:.4f}")
    history.append({"epoch":epoch,"loss":avg,"val_dice":vdice})

    if vdice>best:
        best=vdice
        torch.save(model.state_dict(),BEST_MODEL_PATH)

(OUTPUTS_DIR/"training_history.json").write_text(json.dumps(history,indent=2))
print("Best model saved to:",BEST_MODEL_PATH)
