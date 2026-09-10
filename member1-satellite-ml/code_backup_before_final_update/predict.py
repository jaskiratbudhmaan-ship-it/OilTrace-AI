import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from config import *
from src.preprocessing.preprocess import preprocess_sar_image
from src.models.unet import UNet
from src.utils.polygon import mask_to_polygons, largest_centroid

parser=argparse.ArgumentParser()
parser.add_argument("image")
args=parser.parse_args()

img=preprocess_sar_image(args.image,IMAGE_SIZE)
x=torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

model=UNet().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=DEVICE))
model.eval()

with torch.no_grad():
    prob=torch.sigmoid(model(x))[0,0].cpu().numpy()

mask=(prob>=MASK_THRESHOLD).astype(np.uint8)
confidence=float(prob[mask==1].mean()) if mask.any() else float(prob.max())

polygons=mask_to_polygons(mask)
centroid=largest_centroid(mask)

out=PREDICTIONS_DIR
out.mkdir(parents=True,exist_ok=True)

cv2.imwrite(str(out/"oil_mask.png"),mask*255)

base_img=(img*255).astype(np.uint8)
overlay=cv2.cvtColor(base_img,cv2.COLOR_GRAY2BGR)
overlay[mask==1]=[255,255,255]
cv2.imwrite(str(out/"overlay.png"),overlay)

result={
    "spill_detected": bool(mask.any()),
    "confidence": confidence,
    "oil_pixels": int(mask.sum()),
    "centroid_px": centroid,
    "polygons_px": polygons
}
(out/"result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
print("Outputs saved to:",out)
