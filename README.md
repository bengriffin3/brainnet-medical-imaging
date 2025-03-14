# brainnet-medical-imaging
Comparing Deep Learning Approaches for Medical Image Analysis

This project explores how different deep learning architectures can be used for medical image classification (e.g., brain tumour detection). We will compare traditional CNNs with state-of-the-art models such as Vision Transformers, Variational Autoencoders, and/or leading pre-trained models. Our goal is to understand the trade-offs between these methods (e.g., accuracy, computational efficiency).

## Data Exploration

The `scripts/explore_data.py` script performs initial data analysis on the Brain Tumor MRI Dataset, generating visualizations for:
- Class distribution in training and test sets
- Sample MRI images from the dataset
- Comparison of original and preprocessed images
- Pixel intensity distributions

Run the script with:
```bash
python scripts/explore_data.py
```

Visualizations will be saved in the `outputs` directory.

## CNN Training

Run the training script with:
```bash
python scripts/train_model.py
```

For better performance, it's recommended to run training on a GPU (e.g., using Google Colab). Trained models are saved in the `models` directory.