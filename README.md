# brainnet-medical-imaging

## Comparing Deep Learning Approaches for Medical Image Analysis

This project explores how different deep learning architectures can be used for medical image classification (specifically brain tumor detection). We compare traditional CNNs with more advanced models such as Vision Transformers (ViTs) and CNNs improved by residual connections and topological features.

The goal is to understand trade-offs between these methods in terms of accuracy, computational efficiency, and practical deployment considerations.

---

## Project Structure

- `notebooks/`
  - `01_EDA_and_Preprocessing.ipynb`: Exploratory data analysis (EDA) and preprocessing
  - `02_visualize_baseline_results.ipynb`: Visualizes results for the baseline CNN model
  - `03_compare_model_results.ipynb`: Compares performance across all trained models
- `src/`
  - Core Python modules for data loading, model definitions, training utilities, and visualization
- `results/`
  - Saved training histories and evaluation metrics for each model
- `main.py`
  - Main script to train models
- `requirements.txt`
  - Required Python packages for setting up the environment

---

## Setup

Install dependencies using:

```pip install -r requirements.txt```

Note:
- A GPU is strongly recommended for training, especially for the Vision Transformer models.
- If using Google Colab, select "GPU" as the runtime type.

---

## How to Train Models

To train a model, run the `main.py` script with the following command:

```python main.py --model <model_name> --epochs <num_epochs>```


Where `<model_name>` can be one of:
- `cnn` - Baseline CNN
- `cnn_res` - CNN with Residual Connections
- `cnn_topo` - CNN with Topological Features
- `vit` - Pre-trained Vision Transformer
- `vit_ft` - Fine-tuned Vision Transformer (last transformer block unfrozen)

Example:

```python main.py --model cnn --epochs 10```

The training history and results will be saved automatically in the `results/` directory.

---

## Visualizing Results

After training:

- Baseline CNN results can be visualized in:
  - `notebooks/02_visualize_baseline_results.ipynb`
- Comparison of all models can be visualized in:
  - `notebooks/03_compare_model_results.ipynb`

These notebooks generate plots of:
- Training and validation loss curves
- Validation accuracy curves
- Epoch training times
- Final performance comparisons across models

---

## Summary of Findings

- CNN performed strongly as a lightweight and fast baseline model
- CNN with Residual Blocks and CNN with Topological Features achieved similar accuracies (around 95%), and are expected to show greater benefits on larger datasets or with deeper architectures
- Vision Transformers (ViT), especially when fine-tuned, achieved the highest accuracy (~98.6%) but required significantly longer training times
