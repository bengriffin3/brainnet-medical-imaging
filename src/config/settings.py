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
# The dataset path will be determined at runtime by kagglehub.dataset_download()

# Image settings
CHANNELS = 1  # Grayscale images
MODEL_INPUT_SIZE = (128, 128)  # Size expected by the CNN model

# Training settings
RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 2

# Normalization parameters
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# Model settings
NUM_CLASSES = 4

# Training settings
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Model save path
MODEL_SAVE_PATH = "models/brain_tumor_cnn.pth"
