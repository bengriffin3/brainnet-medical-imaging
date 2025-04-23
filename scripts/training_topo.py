import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: The device to use (CPU or GPU)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for i, (images, topos, labels) in enumerate(train_loader):
        # Move data to GPU if available
        images = images.to(device)
        topos = topos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, topos)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(train_loader)

def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: The device to use (CPU or GPU)
        
    Returns:
        Tuple[float, float]: Average validation loss and accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, topos, labels in val_loader:
            # Move data to GPU if available
            images = images.to(device)
            topos = topos.to(device)
            labels = labels.to(device)
            
            outputs = model(images, topos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / float(total)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    """Train the model for multiple epochs.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        device: The device to use (CPU or GPU)
        learning_rate: Initial learning rate
        
    Returns:
        Dict containing training history (losses and accuracy)
    """
    # Move model to GPU if available
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, accuracy = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch: {epoch} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Accuracy: {accuracy:.2f}%")
    
    return history

def plot_training_history(history: Dict[str, List[float]]) -> None:
    """Plot training history.
    
    Args:
        history: Dictionary containing training metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show() 