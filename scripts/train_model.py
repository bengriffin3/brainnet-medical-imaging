import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import pathlib
import kagglehub
import os

from src.models.cnn import CNNModel
from src.utils.training import train_model, plot_training_history
from src.data.transforms import get_train_transforms, get_val_transforms
from src.config import settings

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset path dynamically
    data_dir = pathlib.Path(kagglehub.dataset_download(settings.KAGGLE_DATASET))
    print(f"Dataset downloaded to: {data_dir}")
    
    train_dir = data_dir.joinpath('Training')
    val_dir = data_dir.joinpath('Testing')  # Using test set as validation
    
    # Create datasets
    train_set = ImageFolder(train_dir, transform=get_train_transforms())
    val_set = ImageFolder(val_dir, transform=get_val_transforms())
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True  # This helps transfer data to GPU faster
    )
    val_loader = DataLoader(
        val_set,
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True  # This helps transfer data to GPU faster
    )
    
    # Initialize model
    model = CNNModel()
    print(model)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=settings.NUM_EPOCHS,
        device=device,
        learning_rate=settings.LEARNING_RATE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(settings.MODEL_SAVE_PATH), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), settings.MODEL_SAVE_PATH)
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 