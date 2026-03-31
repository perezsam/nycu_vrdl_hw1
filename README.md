# NYCU Computer Vision 2026 HW1
**Student ID:** 314540033
**Name:** Samuel Perez

## Introduction
This repository contains the training and inference pipeline for HW1 (100-class Fine-Grained Visual Classification). The final solution utilizes a Multiscale Triple Ensemble strategy. The ensemble combines ResNet-101, ResNeXt-50, and a SOTA-modified ResNeXt-50 featuring Generalized Mean Pooling (GeM), Channel Attention (SE Block), and Focal Loss. Multiscale Test-Time Augmentation (TTA) is used during inference to capture both macro-structures and micro-textures.

## Environment Setup
It is recommended to use a virtual environment (e.g., Conda) with Python 3.10+.

```bash
# Create and activate environment
conda create -n vrdl_hw1 python=3.10 -y
conda activate vrdl_hw1

# Install required dependencies
pip install -r requirements.txt
