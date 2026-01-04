# Paper Title: FP-GTCN: A GCN-TCN-based Soft Sensor with a Flexible Prior Process Graph for Trifluoroethanol Distillation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/xxxx.xxxxx)

Official implementation of **"FP-GTCN: A GCN-TCN-based Soft Sensor with a Flexible Prior Process Graph for Trifluoroethanol Distillation"** published in **Journal Name**.

> **Authors**: Yuting Li,Jie Cheng
> **Institution**: Shandong University/School of Airspace Science and Engineering  
> **Contact**: chjie@sdu.edu.cn

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This repository provides an official implementation of the spatiotemporal soft sensing method FP-GTCN, based on GCN–TCN, for online purity soft sensing tasks in the trifluoroethanol distillation process. 

### Key Features

- Feature 1: Brief description
- Feature 2: Brief description
- Feature 3: Brief description

### Architecture

![Model Architecture](figures/architecture.png)

*Figure 1: Overview of the proposed method/architecture.*

## 🛠️ Installation

### Prerequisites

- Python 3.9.X
- CUDA 11.7 (cu117)

### Environment Setup
This project uses PyTorch Geometric (PyG) with CUDA 11.7, which requires special installation steps. 
Do NOT run `pip install -r requirements.txt` directly. 
Instead, follow these steps:

## Step1:Install PyTorch (with CUDA 11.7) and its related packages first:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```
## Step2:Install PyG dependencies compatible with PyTorch 2.0.1 (CUDA 11.7):
```
pip install torch-scatter==2.1.2+pt20cu117 \
             torch-sparse==0.6.18+pt20cu117 \
             torch-cluster==1.6.3+pt20cu117 \
             torch-spline-conv==1.2.2+pt20cu117 \
   -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```
## Step3:Install the remaining project dependencies (from the `requirements.txt` file):
```
pip install -r requirements.txt
```

## 📊 Dataset

### Dataset Preparation

Describe your dataset and provide download instructions:

```bash
# Download dataset
wget https://example.com/dataset.zip
unzip dataset.zip -d data/

# Preprocess data
python scripts/preprocess.py --input data/raw --output data/processed
```

### Dataset Structure

```
data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🚀 Usage

### Quick Start

```bash
# Train the model
python train.py --config configs/default.yaml

# Evaluate the model
python evaluate.py --checkpoint checkpoints/best_model.pth --data data/test

# Run inference
python inference.py --input sample_input.jpg --output output.jpg
```

### Training

```bash
python train.py \
    --data_path data/processed \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --gpu 0
```

### Evaluation

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_path data/test \
    --metrics accuracy f1 auc
```

### Inference on Custom Data

```bash
python inference.py \
    --input your_data.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output results/
```

## 📈 Results

### Quantitative Results

Performance comparison on benchmark datasets:

| Method | Dataset | Metric 1 | Metric 2 | Metric 3 |
|--------|---------|----------|----------|----------|
| Baseline | Dataset-A | 85.2 | 78.3 | 82.1 |
| Method-B | Dataset-A | 87.5 | 80.1 | 84.3 |
| **Ours** | Dataset-A | **91.3** | **85.7** | **88.9** |

### Qualitative Results

![Qualitative Results](figures/results.png)

*Figure 2: Visualization of results on sample images.*

### Pre-trained Models

Download pre-trained model weights:

- [Model trained on Dataset-A](https://drive.google.com/xxx) (250MB)
- [Model trained on Dataset-B](https://drive.google.com/xxx) (250MB)

## 📁 Project Structure

```
.
├── configs/              # Configuration files
├── data/                 # Dataset directory
├── figures/              # Figures for README
├── models/               # Model definitions
│   ├── __init__.py
│   └── your_model.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
├── scripts/              # Scripts for preprocessing, etc.
├── checkpoints/          # Model checkpoints
├── results/              # Experimental results
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Inference script
├── requirements.txt      # Python dependencies
├── LICENSE               # License file
└── README.md             # This file
```

## 📖 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{lapluie2024yourtitle,
  title={FP-GTCN: A GCN-TCN-based Soft Sensor with a Flexible Prior Process Graph for Trifluoroethanol Distillation},
  author={LiYuting and Chengjie},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX--XXX},
  doi={10.xxxx/xxxxx}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Acknowledge funding sources
- Acknowledge contributors or collaborators
- Mention any code or data sources used

## 📧 Contact

For questions or issues, please:

- Open an issue on GitHub
- Contact: 202437617@mail.sdu.edu
---

**Note**: This repository contains the official implementation. For any questions regarding the paper or code, feel free to reach out.
