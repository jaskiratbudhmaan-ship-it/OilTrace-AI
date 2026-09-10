from pathlib import Path
import cv2, numpy as np, rasterio

SUPPORTED = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

def read_grayscale(path):
    path = Path(path)
    if path.suffix.lower() in {".tif",".tiff"}:
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read {path}")
        img = img.astype(np.float32)
    return img

def robust_normalize(image, p1=1, p99=99):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("No finite values in image")
    fill = float(np.median(image[finite]))
    image = np.where(finite, image, fill)
    lo, hi = np.percentile(image, [p1,p99])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lo, hi)
    return ((image-lo)/(hi-lo)).astype(np.float32)

def preprocess_sar_image(path, size=(256,256)):
    img = robust_normalize(read_grayscale(path))
    img = cv2.resize(img, (size[1],size[0]), interpolation=cv2.INTER_AREA)
    return img

def preprocess_mask(path, size=(256,256)):
    mask = read_grayscale(path)
    mask = np.nan_to_num(mask)
    if mask.max() > 1:
        mask = mask / mask.max()
    mask = cv2.resize(mask, (size[1],size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)
