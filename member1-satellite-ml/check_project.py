import sys

print("\n" + "=" * 64)
print("OILTRACE AI")
print("MEMBER 1 - SATELLITE + ML MODULE")
print("PART 1 PROJECT SETUP CHECK")
print("=" * 64)

errors = []

def safe_import(name, import_statement):
    try:
        module = __import__(import_statement)
        version = getattr(module, "__version__", "installed")
        print(f"✓ {name}: {version}")
        return module
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        print(f"✗ {name}: NOT AVAILABLE")
        return None

print("\n[1] Python")
print("Python version:", sys.version.split()[0])

print("\n[2] Libraries")
torch = safe_import("PyTorch", "torch")
np = safe_import("NumPy", "numpy")
cv2 = safe_import("OpenCV", "cv2")
rasterio = safe_import("Rasterio", "rasterio")
shapely = safe_import("Shapely", "shapely")
geopandas = safe_import("GeoPandas", "geopandas")
albumentations = safe_import("Albumentations", "albumentations")

try:
    from config import (
        BASE_DIR,
        RAW_IMAGES_DIR,
        RAW_MASKS_DIR,
        TRAIN_IMAGES_DIR,
        TRAIN_MASKS_DIR,
        VAL_IMAGES_DIR,
        VAL_MASKS_DIR,
        TEST_IMAGES_DIR,
        TEST_MASKS_DIR,
        MODELS_DIR,
        MASK_OUTPUT_DIR,
        OVERLAY_OUTPUT_DIR,
        METRICS_OUTPUT_DIR,
        DEVICE,
    )
except Exception as exc:
    print("\n✗ Could not import config.py")
    print(exc)
    raise SystemExit(1)

print("\n[3] Compute Device")
print("Selected device:", DEVICE)

if torch is not None:
    print("Apple MPS available:", torch.backends.mps.is_available())
    print("CUDA available     :", torch.cuda.is_available())

print("\n[4] Project Directories")

folders = [
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_MASKS_DIR,
    VAL_IMAGES_DIR,
    VAL_MASKS_DIR,
    TEST_IMAGES_DIR,
    TEST_MASKS_DIR,
    MODELS_DIR,
    MASK_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
]

folders_ok = True

for folder in folders:
    if folder.exists():
        print("✓", folder.relative_to(BASE_DIR))
    else:
        folders_ok = False
        print("✗ Missing:", folder.relative_to(BASE_DIR))

print("\n" + "=" * 64)

if not errors and folders_ok:
    print("✓ PART 1 SETUP SUCCESSFUL")
    print("Project is ready for Part 2: real Sentinel-1 dataset.")
else:
    print("✗ SETUP INCOMPLETE")
    if errors:
        print("\nInstall/fix these libraries:")
        for error in errors:
            print("-", error)

print("=" * 64)
