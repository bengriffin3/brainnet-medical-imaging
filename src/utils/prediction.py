#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, List


import torch
import torch.nn as nn


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_single_image(model: nn.Module, test_set, device: torch.device) -> str:
    """
    Make prediction for a single image.

    Args:
        model: Trained PyTorch model
        test_set: Dataset containing test images
        label_conversion_dict: Dictionary mapping numeric labels to class names
        device: Device to run the model on

    Returns:
        Predicted class name
    """
    
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

    label_conversion_dict = {
            'glioma': 0,
            'meningioma': 1,
            'notumor': 2,
            'pituitary': 3,
            0: 'glioma',
            1: 'meningioma',
            2: 'notumor',
            3: 'pituitary'
        } 
    
    sample_idx = np.random.randint(0, len(test_set))
    image, label = test_set[sample_idx]

    model.eval()
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = label_conversion_dict[pred.item()]

    # Prepare image for display
    image_to_show = image.numpy().squeeze()  # Remove batch dim and convert to numpy
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image_to_show, cmap='gray')
    plt.title(f'True Label: {label_conversion_dict[label]}')
    plt.axis('off')
    plt.show()
    print(f'Predicted Class: {predicted_class}')
    
    return predicted_class



def summary(conf_matrix, class_labels):
    
    print("Summary: \n")
    print("We have \n")

    pred_percent = [0,0,0,0]

    for i in range(4):
        pred_total = sum(conf_matrix[:,i])
        pred_percent[i] = conf_matrix[i,i]/pred_total
        print(f"{conf_matrix[i,i]/pred_total:.2%} accuracy rate for predicting {class_labels[i]}.") 
        print(f"If {class_labels[i]} is predicted, the true class of the predicted image could be:")
        for j in range(4):
            if j != i:
                pred_percent[j] = conf_matrix[j,i]/pred_total
                print(f"{class_labels[j]} with {pred_percent[j]:.2%} chance")
        print("\n")

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
) -> Dict[str, List[float]]:
    """
    Train the model and return training history.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate

    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize lists to track metrics and predictions
    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    all_predictions = []  # To store predictions for all epochs
    all_true_labels = []  # To store true labels for all epochs
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
          if len(batch) == 3:
            images, topos, labels = batch
            images, topos, labels = images.to(device), topos.to(device), labels.to(device)
            outputs = model(images,topos)

          else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device) 
            outputs =  model(images)

          loss = criterion(outputs, labels)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        epoch_predictions = []
        epoch_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
              if len(batch) == 3:
                images, topos, labels = batch
                images, topos, labels = images.to(device), topos.to(device), labels.to(device)
                outputs = model(images,topos)

              else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device) 
                outputs =  model(images)
                
              
              val_loss += criterion(outputs, labels).item()

              # Get predictions
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

              # Store predictions and true labels
              epoch_predictions.extend(predicted.cpu().numpy())
              epoch_true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        # Store history
        train_loss_list.append(train_loss)
        test_loss_list.append(val_loss)
        accuracy_list.append(accuracy)
        all_predictions.append(epoch_predictions)
        all_true_labels.append(epoch_true_labels)

        # Update learning rate
        scheduler.step(val_loss)

        # Calculate and store epoch time
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Print metrics for the current epoch
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'Epoch {epoch+1} took {epoch_time:.2f} seconds')

    # Return the complete history including predictions and labels
    return {
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
        'accuracy_list': accuracy_list,
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'epoch_times': epoch_times
    }

from collections import Counter

# def analyze_predictions(all_predictions, all_true_labels):
#     num_epochs = len(all_predictions)
#     class_counts_per_epoch = []

#     # Count predictions for each epoch
#     for epoch in range(num_epochs):
#         epoch_prediction_counts = Counter(all_predictions[epoch])
#         class_counts_per_epoch.append(epoch_prediction_counts)
#         print(f"Epoch {epoch + 1} Prediction Counts: {epoch_prediction_counts}")

#     # Check if the model is just predicting one or two classes repeatedly
#     flat_predictions = [item for sublist in all_predictions for item in sublist]
#     prediction_counts = Counter(flat_predictions)
#     print(f"\nOverall Prediction Counts Across All Epochs: {prediction_counts}")

#     # Calculate overall accuracy
#     flat_true_labels = [item for sublist in all_true_labels for item in sublist]
#     correct_predictions = sum([1 for pred, true in zip(flat_predictions, flat_true_labels) if pred == true])
#     overall_accuracy = 100 * correct_predictions / len(flat_true_labels)
#     print(f"Overall Accuracy Across All Epochs: {overall_accuracy:.2f}%")

def analyze_predictions(all_predictions, all_true_labels, model_name="Model"):
    """
    Analyze predictions for a model over all epochs.

    Args:
        all_predictions: List of lists of predictions per epoch
        all_true_labels: List of lists of true labels per epoch
        model_name: Name of the model (string) to print in the output
    """
    print(f"\n{'='*60}")
    print(f"Analyzing predictions for: {model_name}")
    print(f"{'='*60}")

    num_epochs = len(all_predictions)
    class_counts_per_epoch = []

    # Count predictions for each epoch
    for epoch in range(num_epochs):
        epoch_prediction_counts = Counter(all_predictions[epoch])
        class_counts_per_epoch.append(epoch_prediction_counts)
        print(f"Epoch {epoch + 1} Prediction Counts: {epoch_prediction_counts}")

    # Overall prediction counts
    flat_predictions = [item for sublist in all_predictions for item in sublist]
    prediction_counts = Counter(flat_predictions)
    print(f"\nOverall Prediction Counts Across All Epochs: {prediction_counts}")

    # Overall accuracy
    flat_true_labels = [item for sublist in all_true_labels for item in sublist]
    correct_predictions = sum([1 for pred, true in zip(flat_predictions, flat_true_labels) if pred == true])
    overall_accuracy = 100 * correct_predictions / len(flat_true_labels)
    print(f"Overall Accuracy Across All Epochs: {overall_accuracy:.2f}%")
    print("\n")

def load_history(history_filename, results_dir='../results'):
    """
    Load only the training history.

    Args:
        history_filename: Name of the history file (e.g., 'cnn_history_10_epochs.pkl')
        results_dir: Directory where the history is saved

    Returns:
        history: Loaded training history dictionary
    """
    history_path = os.path.join(results_dir, history_filename)
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    return history