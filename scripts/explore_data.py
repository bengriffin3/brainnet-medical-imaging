"""
Exploratory data analysis script for the brain tumor classification project.
"""

import sys
import gc
import torch
from pathlib import Path
from src.data.dataset import get_data_loaders, get_class_distribution
from src.data.transforms import get_train_transforms
from src.utils.visualization import (
    plot_class_distribution, 
    plot_pixel_distribution,
    plot_sample_images,
    plot_original_vs_transformed
)
import matplotlib.pyplot as plt

def main():
    try:
        # Create output directory for plots
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Get data loaders
        print("Loading data...")
        train_loader, test_loader = get_data_loaders()
        
        # Get class distributions
        print("\nCalculating class distributions...")
        train_counts = get_class_distribution(train_loader.dataset)
        test_counts = get_class_distribution(test_loader.dataset)
        
        # Print class counts
        print("\nTraining set class distribution:")
        for class_name, count in train_counts.items():
            print(f"{class_name}: {count}")
        
        print("\nTest set class distribution:")
        for class_name, count in test_counts.items():
            print(f"{class_name}: {count}")
        
        # Plot and save class distribution
        print("\nPlotting class distribution...")
        plot_class_distribution(train_counts, test_counts)
        plt.savefig(output_dir / "class_distribution.png")
        plt.close()
        
        # Plot and save sample images
        print("\nPlotting sample images...")
        plot_sample_images(train_loader)
        plt.savefig(output_dir / "sample_images_grid.png")
        plt.close()
        
        # Plot and save original vs transformed images
        print("\nPlotting original vs transformed images...")
        plot_original_vs_transformed(train_loader.dataset, get_train_transforms(), num_samples=6)
        plt.savefig(output_dir / "transform_comparison.png")
        plt.close()
        
        # Get and plot a sample image pixel distribution
        print("\nPlotting sample image pixel distribution...")
        sample_image, _ = next(iter(train_loader))
        plot_pixel_distribution(sample_image[0])  # Take first image from batch
        plt.savefig(output_dir / "pixel_distribution.png")
        plt.close()
        
        # Clean up memory
        del train_loader, test_loader, sample_image
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\nAnalysis complete! Plots have been saved to the 'outputs' directory:")
        print("1. class_distribution.png - Distribution of classes in train/test sets")
        print("2. sample_images_grid.png - Grid of sample images from the dataset")
        print("3. transform_comparison.png - Comparison of original vs transformed images")
        print("4. pixel_distribution.png - Distribution of pixel intensities")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 