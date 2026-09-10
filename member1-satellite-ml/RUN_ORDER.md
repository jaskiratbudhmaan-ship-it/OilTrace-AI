# RUN ORDER

1. `./setup_mac.sh`
2. `source .venv/bin/activate`
3. `python scripts/download_dataset.py`
4. `brew install unar`  (only if extractor is missing)
5. `python scripts/extract_dataset.py`
6. `python scripts/prepare_dataset.py`
7. `python scripts/check_dataset.py`
8. `python scripts/test_dataset_loader.py`
9. `python train.py`
10. `python evaluate.py`
11. `python predict.py data/processed/test/images/test_0000.png`

Do not push the dataset or model weights to GitHub.
