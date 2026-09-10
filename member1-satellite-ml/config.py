from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
KAGGLE_IMAGES_DIR = EXTRACTED_DIR / "images"
KAGGLE_MASKS_DIR = EXTRACTED_DIR / "masks"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_IMAGES_DIR = PROCESSED_DIR / "train" / "images"
TRAIN_MASKS_DIR = PROCESSED_DIR / "train" / "masks"
VAL_IMAGES_DIR = PROCESSED_DIR / "val" / "images"
VAL_MASKS_DIR = PROCESSED_DIR / "val" / "masks"
TEST_IMAGES_DIR = PROCESSED_DIR / "test" / "images"
TEST_MASKS_DIR = PROCESSED_DIR / "test" / "masks"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_unet_model.pt"
LAST_MODEL_PATH = MODELS_DIR / "last_unet_model.pt"

OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CURVES_DIR = OUTPUTS_DIR / "curves"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
MASK_THRESHOLD = 0.5
UNET_BASE_CHANNELS = 32

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

for d in [MODELS_DIR, METRICS_OUTPUT_DIR, PREDICTIONS_DIR, CURVES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
