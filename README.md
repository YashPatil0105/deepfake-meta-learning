# Deepfake Detection using Meta-Learning

A comprehensive framework for detecting deepfakes using few-shot meta-learning approaches. This project implements multiple meta-learning algorithms (ProtoNet, RelationNet, MAML, Reptile) trained on deepfake datasets to build robust detectors that can generalize with minimal examples.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Deepfakes pose significant challenges to media authenticity and security. Traditional deep learning approaches require large amounts of labeled data. This project tackles the problem using **few-shot meta-learning**, enabling the model to detect new types of deepfakes with minimal training examples.

### Key Algorithms

- **ProtoNet**: Prototype Networks for few-shot learning
- **RelationNet**: Relation Networks for metric learning
- **MAML**: Model-Agnostic Meta-Learning
- **Reptile**: First-order meta-learning algorithm

## Features

- ✅ Multi-dataset support (FaceForensics++, CelebDF, DFDC)
- ✅ Multiple meta-learning algorithms
- ✅ Comprehensive data preprocessing pipeline
- ✅ Few-shot task generation
- ✅ Temporal analysis support
- ✅ Cross-dataset evaluation
- ✅ Detailed experiment tracking and visualization
- ✅ Ablation studies framework

## Project Structure

```
deepfake-meta-learning/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
│
├── data/                             # Data storage
│   ├── raw/                          # Raw video datasets
│   │   ├── faceforensics++/         # FaceForensics++ dataset
│   │   │   ├── videos/              # Video files
│   │   │   └── metadata/            # Metadata and labels
│   │   ├── celebdf/                 # CelebDF dataset
│   │   │   ├── videos/
│   │   │   └── metadata/
│   │   └── dfdc/                    # DFDC dataset
│   │       ├── videos/
│   │       └── metadata/
│   │
│   ├── processed/                    # Processed data
│   │   ├── frames/                  # Extracted frames
│   │   │   ├── train/               # Training frames
│   │   │   ├── val/                 # Validation frames
│   │   │   └── test/                # Test frames
│   │   ├── faces/                   # Detected face regions
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── embeddings/              # Feature embeddings
│   │       ├── resnet18/            # ResNet-18 embeddings
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── test/
│   │       └── temporal/            # Temporal embeddings (optional)
│   │
│   └── splits/                       # Task definitions
│       ├── meta_train_tasks.json     # Meta-training tasks
│       ├── meta_val_tasks.json       # Meta-validation tasks
│       └── meta_test_tasks.json      # Meta-testing tasks
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb         # Data analysis
│   ├── 02_preprocessing_pipeline.ipynb      # Preprocessing workflow
│   ├── 03_feature_extraction.ipynb          # Feature generation
│   ├── 04_baseline_cnn.ipynb                # Baseline CNN model
│   ├── 05_few_shot_task_builder.ipynb       # Task generation
│   ├── 06_protonet.ipynb                    # ProtoNet experiments
│   ├── 07_relationnet.ipynb                 # RelationNet experiments
│   ├── 08_reptile.ipynb                     # Reptile experiments
│   ├── 09_maml.ipynb                        # MAML experiments
│   ├── 10_ablation_studies.ipynb            # Ablation studies
│   └── 11_result_visualization.ipynb        # Results visualization
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── data/                         # Data processing modules
│   │   ├── __init__.py
│   │   ├── video_loader.py           # Load video files
│   │   ├── frame_extractor.py        # Extract frames from videos
│   │   ├── face_detector.py          # Detect faces in frames
│   │   ├── dataset_builder.py        # Build datasets
│   │   └── task_sampler.py           # Sample few-shot tasks
│   │
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── backbones/                # Feature extractors
│   │   │   ├── __init__.py
│   │   │   └── resnet18.py           # ResNet-18 backbone
│   │   ├── meta_learning/            # Meta-learning algorithms
│   │   │   ├── __init__.py
│   │   │   ├── protonet.py           # Prototype Networks
│   │   │   ├── relationnet.py        # Relation Networks
│   │   │   ├── maml.py               # MAML
│   │   │   └── reptile.py            # Reptile
│   │   └── temporal/                 # Temporal models
│   │       ├── __init__.py
│   │       └── cnn_lstm.py           # CNN-LSTM architecture
│   │
│   ├── training/                     # Training scripts
│   │   ├── __init__.py
│   │   ├── train_baseline.py         # Baseline CNN training
│   │   ├── train_meta.py             # Meta-learning training
│   │   └── evaluate.py               # Evaluation utilities
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── losses.py                 # Loss functions
│   │   ├── visualization.py          # Plotting utilities
│   │   └── logger.py                 # Logging utilities
│   │
│   └── configs/                      # Configuration files
│       ├── dataset.yaml              # Dataset configurations
│       ├── preprocessing.yaml        # Preprocessing configs
│       └── meta_learning.yaml        # Meta-learning configs
│
├── experiments/                      # Experiment results
│   ├── baseline_results/             # Baseline model results
│   ├── protonet_results/             # ProtoNet results
│   ├── relationnet_results/          # RelationNet results
│   ├── reptile_results/              # Reptile results
│   └── maml_results/                 # MAML results
│
├── figures/                          # Visualizations and plots
│   ├── architecture_diagram.png
│   ├── fft_analysis.png
│   ├── temporal_analysis.png
│   ├── accuracy_vs_shot.png
│   └── cross_dataset_results.png
│
├── paper/                            # Paper and documentation
│   ├── outline.md                    # Paper outline
│   ├── methodology.md                # Methodology section
│   ├── experiments.md                # Experiments section
│   ├── results.md                    # Results section
│   └── references.bib                # Bibliography
│
└── scripts/                          # Automation scripts
    ├── preprocess_all.sh             # Preprocess all datasets
    ├── extract_embeddings.sh         # Extract embeddings
    └── run_all_experiments.sh        # Run all experiments

```

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- GPU (NVIDIA) recommended for faster training (CUDA 11.0+)
- At least 100GB disk space for datasets

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/deepfake-meta-learning.git
cd deepfake-meta-learning
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision version: {torchvision.__version__}')"
```
