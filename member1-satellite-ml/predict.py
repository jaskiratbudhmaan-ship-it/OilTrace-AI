import argparse,json,cv2,numpy as np,torch
from pathlib import Path
from config import *
from src.models.unet import UNet
from src.utils.polygon import clean_binary_mask,mask_to_polygons,centroid_from_mask

p=argparse.ArgumentParser(); p.add_argument("image"); p.add_argument("--threshold",type=float,default=MASK_THRESHOLD)
a=p.parse_args(); inp=Path(a.image)
img=cv2.imread(str(inp),0)
if img is None: raise ValueError(f"Cannot read {inp}")
img=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
x=torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
model=UNet(base_channels=UNET_BASE_CHANNELS).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=DEVICE)); model.eval()
with torch.no_grad(): prob=torch.sigmoid(model(x))[0,0].cpu().numpy()
mask=clean_binary_mask((prob>=a.threshold).astype(np.uint8),20)
conf=float(prob[mask==1].mean()) if mask.any() else float(prob.max())
polys=mask_to_polygons(mask,20); cent=centroid_from_mask(mask)
out=PREDICTIONS_DIR/inp.stem; out.mkdir(parents=True,exist_ok=True)
orig=(img*255).astype(np.uint8); cv2.imwrite(str(out/"original.png"),orig)
cv2.imwrite(str(out/"probability_map.png"),(prob*255).clip(0,255).astype(np.uint8))
cv2.imwrite(str(out/"oil_mask.png"),mask*255)
overlay=cv2.cvtColor(orig,cv2.COLOR_GRAY2BGR); overlay[mask==1]=(0,0,255)
cv2.imwrite(str(out/"overlay.png"),overlay)
res={"spill_detected":bool(mask.any()),"confidence":conf,"oil_pixels":int(mask.sum()),
     "oil_fraction":float(mask.mean()),"centroid_px":cent,"polygons_px":polys,
     "note":"Kaggle chips are not georeferenced in this pipeline, so centroid/polygons are pixel coordinates."}
(out/"result.json").write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2)); print("Saved:",out)
