# OilTrace AI — Member 1 Complete

Member 1 scope:

Sentinel-1 SAR image → preprocessing → real dataset → U-Net segmentation → training → evaluation → oil mask → polygon → confidence.

## Commands

### 1. Setup
```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 2. Download real dataset
```bash
source .venv/bin/activate
python scripts/download_dataset.py
```

### 3. Extract dataset
If needed:
```bash
brew install unar
```

Then:
```bash
python scripts/extract_dataset.py
```

### 4. Prepare dataset
```bash
python scripts/prepare_dataset.py
python scripts/check_dataset.py
python scripts/test_dataset_loader.py
```

### 5. Train
```bash
python train.py
```

### 6. Evaluate
```bash
python evaluate.py
```

### 7. Predict on a processed/test SAR image
```bash
python predict.py data/processed/test/images/test_0000.png
```

Prediction outputs appear in:
```text
outputs/predictions/
```

## Real dataset
Zenodo record: 4672426
Oil Spill Segmentation dataset, Sentinel-1A GRD VV, Gulf of Mexico.
