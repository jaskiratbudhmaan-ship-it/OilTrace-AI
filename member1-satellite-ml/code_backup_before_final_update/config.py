from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
EXTRACTED_DIR = DATA_DIR / "extracted"

RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
RAW_MASKS_DIR = DATA_DIR / "raw" / "masks"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_IMAGES_DIR = PROCESSED_DIR / "train" / "images"
TRAIN_MASKS_DIR = PROCESSED_DIR / "train" / "masks"
VAL_IMAGES_DIR = PROCESSED_DIR / "val" / "images"
VAL_MASKS_DIR = PROCESSED_DIR / "val" / "masks"
TEST_IMAGES_DIR = PROCESSED_DIR / "test" / "images"
TEST_MASKS_DIR = PROCESSED_DIR / "test" / "masks"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_unet_model.pt"

OUTPUTS_DIR = BASE_DIR / "outputs"
MASK_OUTPUT_DIR = OUTPUTS_DIR / "masks"
OVERLAY_OUTPUT_DIR = OUTPUTS_DIR / "overlays"
METRICS_OUTPUT_DIR = OUTPUTS_DIR / "metrics"
DATASET_PREVIEW_DIR = OUTPUTS_DIR / "dataset_preview"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MASK_THRESHOLD = 0.5
RANDOM_SEED = 42

ZENODO_RECORD = "4672426"
ZENODO_ARCHIVE_NAME = "Radar_data.rar"
ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/4672426/files/Radar_data.rar?download=1"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DIRECTORIES = [
    DOWNLOADS_DIR, EXTRACTED_DIR, RAW_IMAGES_DIR, RAW_MASKS_DIR,
    TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR,
    TEST_IMAGES_DIR, TEST_MASKS_DIR, MODELS_DIR, MASK_OUTPUT_DIR,
    OVERLAY_OUTPUT_DIR, METRICS_OUTPUT_DIR, DATASET_PREVIEW_DIR,
    PREDICTIONS_DIR
]

def create_project_directories():
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

create_project_directories()

if __name__ == "__main__":
    print("="*60)
    print("OilTrace AI - Member 1")
    print("="*60)
    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Dataset record:", ZENODO_RECORD)
