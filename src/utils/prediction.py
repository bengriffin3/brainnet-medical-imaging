#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# In[2]:





# In[7]:


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


# In[4]:


def conf_matrix(model:nn.Module, test_set):
    
    device = torch.device('cuda')

    true_labels = []
    predicted_labels = []

    model.eval()
    with torch.no_grad():
        for image, label in test_set:
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
        
            true_labels.append(label)  # Keep as numeric
            predicted_labels.append(pred.item())  # Keep as numeric

    # Convert numeric labels to class names for the confusion matrix
    true_labels_names = [label_conversion_dict[l] for l in true_labels]
    predicted_labels_names = [label_conversion_dict[l] for l in predicted_labels]

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels_names, predicted_labels_names, labels=labels)

    # Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Return
    return conf_matrix


# In[6]:


def summary(conf_matrix):
    
    print("Summary: \n")
    print("We have \n")

    pred_percent = [0,0,0,0]

    for i in range(4):
        pred_total = sum(conf_matrix[:,i])
        pred_percent[i] = conf_matrix[i,i]/pred_total
        print(f"{conf_matrix[i,i]/pred_total:.2%} accuracy rate for predicting {labels[i]}.") 
        print(f"If {labels[i]} is predicted, the true class of the predicted image could be:")
        for j in range(4):
            if j != i:
                pred_percent[j] = conf_matrix[j,i]/pred_total
                print(f"{labels[j]} with {pred_percent[j]:.2%} chance")
        print("\n")


# In[ ]:




