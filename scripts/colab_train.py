"""
Run this in Google Colab to train the model with GPU access.

Steps:
1. Upload this entire project to your Google Drive
2. Mount your Google Drive in Colab
3. Run this script
"""

import os
import sys
from pathlib import Path

def setup_colab():
    """Setup the Colab environment."""
    # Install required packages
    os.system('pip install kagglehub')

    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Add project root to path
    project_path = "/content/drive/MyDrive/path/to/your/project"  # Update this!
    sys.path.append(project_path)
    
    # Verify GPU
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    return project_path

if __name__ == "__main__":
    # Setup environment
    project_path = setup_colab()
    os.chdir(project_path)
    
    # Import and run training
    from scripts.train_model import main
    main() 