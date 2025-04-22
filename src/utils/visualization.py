"""
Visualization utilities for the brain tumor classification project.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from PIL import Image
import random

def plot_class_distribution(train_values, test_values, class_labels, ax):
    """
    Plot the class distribution comparison between train and test sets.
    
    Args:
        train_counts (dict): Training set class counts
        test_counts (dict): Test set class counts
    """
    bar_width = 0.35
    index = np.arange(len(class_labels))
    
    # Create bars with specific colors
    bars_train = ax.bar(index, train_values, bar_width, label='Training', color='skyblue')
    bars_test = ax.bar(index + bar_width, test_values, bar_width, label='Testing', color='lightgreen')
    
    # Add value labels on top of bars
    for bars in [bars_train, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution Comparison")
    ax.set_xticks(index + bar_width/2)
    ax.set_ylim(0, max(max(train_values), max(test_values)) + 300)
    ax.set_xticklabels(class_labels, rotation=45)
    ax.legend()

def plot_pixel_distribution(image_tensor, ax):
    """
    Plot the distribution of pixel intensities in an image.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor
    """

    pixel_values = image_tensor.numpy().flatten()
    ax.hist(pixel_values, bins=50, density=True)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Pixel Intensities')

def plot_sample_images(dataloader, nrows=3, ncols=3):
    """
    Display a grid of sample images from the dataset with their labels.
    
    Args:
        dataloader: PyTorch DataLoader
        nrows: Number of rows in the grid
        ncols: Number of columns in the grid
    """
    # Get enough images to fill the grid
    total_images = nrows * ncols
    all_images = []
    all_labels = []
    
    # Collect enough images
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
        if len(torch.cat(all_images)) >= total_images:
            break
    
    # Concatenate all collected images and labels
    images = torch.cat(all_images)
    labels = torch.cat(all_labels)
    
    # Select the required number of images
    images = images[:total_images]
    labels = labels[:total_images]
    
    # Create figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    
    # Plot each image
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            img = images[idx].numpy().transpose((1, 2, 0))
            label = labels[idx]
            
            # Plot image
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            
            # Add class label as title with larger font
            class_name = dataloader.dataset.classes[label]
            axes[i, j].set_title(class_name, pad=10, fontsize=18)
    
    plt.tight_layout()

def plot_original_vs_transformed(dataset, transform, num_samples=6):
    """
    Display original and transformed versions of images side by side.
    
    Args:
        dataset: Original dataset (ImageFolder)
        transform: Transformation to apply
        num_samples: Number of samples to show (default 6 pairs)
    """
    # Get random indices
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    # Create figure with 3x4 grid (3 rows, 4 columns)
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 6))
    
    # Iterate over samples to fill the entire grid
    for i in range(num_samples):
        row = i // 2    # Each row shows 2 pairs (4 images)
        col_start = (i % 2) * 2  # Start column for each pair
        
        # Get original image
        img_path, _ = dataset.imgs[sample_indices[i]]
        img_orig = Image.open(img_path).convert('RGB')
        img_orig_np = np.array(img_orig)
        orig_shape = img_orig_np.shape
        orig_mean = img_orig_np.mean()
        orig_std = img_orig_np.std()
        orig_type = type(img_orig)
        
        # Apply transformation
        img_trans = transform(img_orig)
        trans_shape = tuple(img_trans.shape)
        trans_mean = img_trans.mean().item()
        trans_std = img_trans.std().item()
        trans_type = type(img_trans)
        
        # Display original image
        axes[row, col_start].imshow(img_orig)
        axes[row, col_start].set_title(
            # f"Original: {orig_shape}\nMean: {orig_mean:.2f}\nStd: {orig_std:.2f}",
            f"Original: {orig_shape}\nMean: {orig_mean:.2f}, Std: {orig_std:.2f}\nType: {orig_type.__name__}",
            fontsize=9
        )
        axes[row, col_start].axis('off')
        
        # Display transformed image
        axes[row, col_start + 1].imshow(img_trans.squeeze(), cmap="gray")
        axes[row, col_start + 1].set_title(
            # f"Transformed: {trans_shape}\nMean: {trans_mean:.2f}\nStd: {trans_std:.2f}",
            f"Transformed: {trans_shape}\nMean: {trans_mean:.2f}, Std: {trans_std:.2f}\nType: {trans_type.__name__}",
            fontsize=9
        )
        axes[row, col_start + 1].axis('off')
    
    plt.tight_layout()


def plot_sample_by_class(dataloader):
    """
    Display one sample image from each class.
    
    Args:
        dataloader: PyTorch DataLoader
    """
    # Get class names
    class_names = dataloader.dataset.classes
    
    # Create dictionary to store one image per class
    class_images = {class_name: None for class_name in class_names}
    
    # Get one image per class
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            class_name = class_names[label]
            if class_images[class_name] is None:
                class_images[class_name] = img
        if all(img is not None for img in class_images.values()):
            break
    
    # Create subplot for each class
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))
    for ax, (class_name, img) in zip(axes, class_images.items()):
        img = img.numpy().transpose((1, 2, 0))
        ax.imshow(img, cmap='gray')
        ax.set_title(class_name)
        ax.axis('off')
    plt.tight_layout()

from typing import Dict, List

def plot_training_history(history: Dict[str, List[float]]):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(history['train_loss_list'], label='Train Loss')
    ax1.plot(history['test_loss_list'], label='Validation Loss')  # Change 'val_loss' to 'test_loss_list'
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=12)

    # Plot accuracy
    ax2.plot(history['accuracy_list'], label='Validation Accuracy')  # Change 'accuracy' to 'accuracy_list'
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

# Function to plot the epoch times for each model
def plot_epoch_times(epoch_times, model_name, ax):

    # Set epochs based on the length of the epoch times
    epochs = range(1, len(epoch_times) + 1)
    ax.plot(epochs, epoch_times, label=f'{model_name} Epoch Time', color='blue')
    ax.set_title(f"Epoch Times Over Epochs ({model_name})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Time (seconds)")
    ax.legend()

# Function to create the plots
def plot_accuracy_results(train_loss, test_loss, accuracy, model_name, ax1, ax2):

    # Set epochs based on the length of the training loss data
    epochs = range(1, len(train_loss) + 1)
    ax1.plot(epochs, train_loss, label=f'{model_name} Train Loss')
    ax1.plot(epochs, test_loss, label=f'{model_name} Test Loss')

    ax2.plot(epochs, accuracy, label=f'{model_name} Accuracy')

    ax1.set_title(f"Training and Validation Loss Over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title(f"Validation Accuracy Over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()