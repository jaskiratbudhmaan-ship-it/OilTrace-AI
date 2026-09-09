# OilTrace AI — Member 1 Satellite + ML Module

This folder contains **Part 1** of the Member 1 implementation for the marine oil-spill detection project.

## Member 1 responsibility

The Member 1 pipeline will eventually be:

**Sentinel-1 SAR image → preprocessing → dataset loader → U-Net semantic segmentation → oil-spill mask → polygon → confidence**

Part 1 only prepares the development environment and project structure.

## Folder structure

```text
member1-satellite-ml/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── masks/
│   └── processed/
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       │   ├── images/
│       │   └── masks/
│       └── test/
│           ├── images/
│           └── masks/
├── src/
│   ├── preprocessing/
│   ├── dataset/
│   ├── models/
│   └── utils/
├── scripts/
├── models/
├── outputs/
│   ├── masks/
│   ├── overlays/
│   └── metrics/
├── config.py
├── requirements.txt
├── check_project.py
├── .gitignore
└── README.md
```

## Setup on macOS

Open this folder in VS Code and open **Terminal → New Terminal**.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the configuration:

```bash
python config.py
```

Verify the complete setup:

```bash
python check_project.py
```

When everything is correct, the final line should say:

```text
✓ PART 1 SETUP SUCCESSFUL
Project is ready for Part 2: real Sentinel-1 dataset.
```

## Git commands

From the root `OilTrace-AI` repository:

```bash
git add .
git commit -m "Complete Member 1 Part 1 project setup"
git push origin main
```

## Important

Large satellite datasets, processed images, generated outputs and trained model weights are intentionally excluded from Git through `.gitignore`.

## Next

**Part 2:** Real Sentinel-1 oil-spill dataset preparation and preprocessing.
