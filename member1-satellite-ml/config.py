from pathlib import Path
import torch

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_MASKS_DIR = RAW_DATA_DIR / "masks"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DATA_DIR / "train"
TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
TRAIN_MASKS_DIR = TRAIN_DIR / "masks"

VAL_DIR = PROCESSED_DATA_DIR / "val"
VAL_IMAGES_DIR = VAL_DIR / "images"
VAL_MASKS_DIR = VAL_DIR / "masks"

TEST_DIR = PROCESSED_DATA_DIR / "test"
TEST_IMAGES_DIR = TEST_DIR / "images"
TEST_MASKS_DIR = TEST_DIR / "masks"

# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_unet_model.pt"

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUTS_DIR = BASE_DIR / "outputs"
MASK_OUTPUT_DIR = OUTPUTS_DIR / "masks"
OVERLAY_OUTPUT_DIR = OUTPUTS_DIR / "overlays"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"

# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30

# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# SEGMENTATION SETTINGS
# ============================================================

MASK_THRESHOLD = 0.5

# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42

# ============================================================
# DEVICE SELECTION
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ============================================================
# DIRECTORY CREATION
# ============================================================

DIRECTORIES = [
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

def create_project_directories():
    """Create all project directories required by the Member 1 module."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

create_project_directories()

if __name__ == "__main__":
    print("=" * 58)
    print("OilTrace AI - Member 1 Configuration")
    print("=" * 58)
    print("Project directory :", BASE_DIR)
    print("Image size        :", IMAGE_SIZE)
    print("Batch size        :", BATCH_SIZE)
    print("Learning rate     :", LEARNING_RATE)
    print("Epochs            :", NUM_EPOCHS)
    print("Mask threshold    :", MASK_THRESHOLD)
    print("Device            :", DEVICE)
    print("=" * 58)
