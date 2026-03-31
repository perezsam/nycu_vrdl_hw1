# NYCU Computer Vision 2026 HW1
**Student ID:** 314540033

**Name:** Samuel Perez 培雷斯

## Introduction
This repository contains the training and inference pipeline for HW1 (100-class Fine-Grained Visual Classification). The final solution utilizes an Architecturally Heterogeneous Ensemble. The ensemble combines ResNet-101, ResNeXt-50, and a SOTA-modified ResNeXt-50 featuring Generalized Mean Pooling (GeM), Channel Attention (SE Block), and Focal Loss. Multiscale Test-Time Augmentation (TTA) is used during inference to capture both macro-structures and micro-textures, effectively mitigating localized visual traps.

## Environment Setup
It is recommended to use a virtual environment (e.g., Conda) with Python 3.10+.

` ` `bash
# Create and activate environment
conda create -n vrdl_hw1 python=3.10 -y
conda activate vrdl_hw1

# Install required dependencies
pip install -r requirements.txt
` ` `

## Usage
### Training
To reproduce the ensemble models, ensure your data is located in `./data/train/` and run the three training scripts sequentially. Each script will save a `.pth` model weights file to the root directory.

` ` `bash
python train_model_A.py
python train_model_B.py
python train_model_D.py
` ` `

### Inference
Ensure `model_A_resnet101.pth`, `model_B_resnext50.pth`, and `model_D_resnext50_sota.pth` are located in the root directory alongside the `./data/test/` folder. 

Run the multiscale ensemble inference script:

` ` `bash
python inference.py
` ` `
This will output the final `prediction.csv` file required for CodaBench evaluation.

## Performance Snapshot
*(See snapshot.png for the final CodaBench leaderboard score of ~0.97)*

![Leaderboard Score](./snapshot.png)
