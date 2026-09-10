from pathlib import Path
import sys, random, json, cv2, numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import *
from src.preprocessing.preprocess import SUPPORTED, preprocess_sar_image, preprocess_mask

MASK_WORDS = ("mask","label","annotation","groundtruth","ground_truth","gt")

def is_img(p): return p.is_file() and p.suffix.lower() in SUPPORTED
def looks_mask(p): return any(w in str(p).lower() for w in MASK_WORDS)

def key(p):
    s = p.stem.lower()
    for t in ["_mask","-mask","_label","-label","_gt","-gt","_groundtruth","-groundtruth"]:
        s=s.replace(t,"")
    return "".join(c for c in s if c.isalnum())

def discover_pairs(root):
    files=[p for p in Path(root).rglob("*") if is_img(p)]
    masks=[p for p in files if looks_mask(p)]
    images=[p for p in files if not looks_mask(p)]
    mmap={}
    for m in masks: mmap.setdefault(key(m),[]).append(m)
    pairs=[]
    for im in images:
        k=key(im)
        cand=mmap.get(k,[])
        if cand: pairs.append((im,cand[0]))
    return pairs

def save_pair(src_img, src_mask, out_img, out_mask):
    img=preprocess_sar_image(src_img, IMAGE_SIZE)
    mask=preprocess_mask(src_mask, IMAGE_SIZE)
    cv2.imwrite(str(out_img), (img*255).astype(np.uint8))
    cv2.imwrite(str(out_mask), (mask*255).astype(np.uint8))

def main():
    pairs=discover_pairs(EXTRACTED_DIR)
    if not pairs:
        print("No image-mask pairs detected.")
        print("Send a screenshot of data/extracted if this happens.")
        raise SystemExit(1)

    random.Random(RANDOM_SEED).shuffle(pairs)
    n=len(pairs)
    ntrain=max(1,int(n*TRAIN_SPLIT))
    nval=max(1,int(n*VAL_SPLIT)) if n>=3 else 0
    if ntrain+nval>=n: ntrain=max(1,n-2); nval=1
    splits={
        "train": pairs[:ntrain],
        "val": pairs[ntrain:ntrain+nval],
        "test": pairs[ntrain+nval:]
    }

    dirs={
        "train":(TRAIN_IMAGES_DIR,TRAIN_MASKS_DIR),
        "val":(VAL_IMAGES_DIR,VAL_MASKS_DIR),
        "test":(TEST_IMAGES_DIR,TEST_MASKS_DIR)
    }

    report={}
    for name, items in splits.items():
        idir,mdir=dirs[name]
        for p in list(idir.glob("*.png"))+list(mdir.glob("*.png")): p.unlink()
        for i,(im,mk) in enumerate(items):
            fn=f"{name}_{i:04d}.png"
            save_pair(im,mk,idir/fn,mdir/fn)
        report[name]=len(items)

    (OUTPUTS_DIR/"dataset_report.json").write_text(json.dumps(report,indent=2))
    print("Prepared:", report)

if __name__=="__main__": main()
