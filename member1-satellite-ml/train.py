import json, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from src.dataset.oil_dataset import OilSpillDataset
from src.models.unet import UNet
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import batch_confusion, metrics_from_counts

train_ds=OilSpillDataset(TRAIN_IMAGES_DIR,TRAIN_MASKS_DIR)
val_ds=OilSpillDataset(VAL_IMAGES_DIR,VAL_MASKS_DIR)
train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

model=UNet(base_channels=UNET_BASE_CHANNELS).to(DEVICE)
crit=BCEDiceLoss()
opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2)

best=-1; stale=0; hist=[]
for epoch in range(1,NUM_EPOCHS+1):
    model.train(); run=0
    for b in tqdm(train_loader,desc=f"Epoch {epoch}/{NUM_EPOCHS} train"):
        x=b["image"].to(DEVICE); y=b["mask"].to(DEVICE)
        opt.zero_grad(set_to_none=True); z=model(x); loss=crit(z,y)
        loss.backward(); opt.step(); run+=loss.item()
    tr=run/max(1,len(train_loader))

    model.eval(); vl=0; tp=fp=fn=tn=0
    with torch.no_grad():
        for b in tqdm(val_loader,desc=f"Epoch {epoch}/{NUM_EPOCHS} val"):
            x=b["image"].to(DEVICE); y=b["mask"].to(DEVICE)
            z=model(x); vl+=crit(z,y).item()
            a,b1,c,d=batch_confusion(z,y,MASK_THRESHOLD)
            tp+=a; fp+=b1; fn+=c; tn+=d
    vm=metrics_from_counts(tp,fp,fn,tn); vd=vm["dice"]; sched.step(vd)
    row={"epoch":epoch,"train_loss":tr,"val_loss":vl/max(1,len(val_loader)),**vm}
    hist.append(row)
    print(row)
    torch.save(model.state_dict(),LAST_MODEL_PATH)
    if vd>best:
        best=vd; stale=0; torch.save(model.state_dict(),BEST_MODEL_PATH); print("✓ saved best model")
    else:
        stale+=1
        if stale>=EARLY_STOPPING_PATIENCE:
            print("Early stopping"); break

(OUTPUTS_DIR/"training_history.json").write_text(json.dumps(hist,indent=2))
print("Best model:",BEST_MODEL_PATH)
