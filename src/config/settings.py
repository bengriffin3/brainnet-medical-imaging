"""
Configuration settings for the brain tumor classification project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"

# Image settings
IMG_SIZE = (224, 224)  # Standard size for many CNN architectures
CHANNELS = 1  # Grayscale images

# Training settings
RANDOM_SEED = 42
BATCH_SIZE = 8  # Reduced from 32 to handle memory better
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

# Normalization parameters
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
