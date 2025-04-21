import torch
import torch.nn as nn
from collections import Counter

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate_one_epoch(model, dataloader, criterion, device):
    """Validate the model for one epoch."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    epoch_predictions = []
    epoch_true_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and true labels
            epoch_predictions.extend(predicted.cpu().numpy())
            epoch_true_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_test_loss, accuracy, epoch_predictions, epoch_true_labels

def run_training(model, train_loader, test_loader, optimizer, scheduler, criterion, num_epochs, device, fine_tuning=False):
    """Run the training and evaluation process for the model."""
    train_loss_list = []
    test_loss_list = []
    accuracy_list = []
    all_predictions = []
    all_true_labels = []

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss_list.append(train_loss)

        # Validate the model
        val_loss, accuracy, epoch_predictions, epoch_true_labels = validate_one_epoch(model, test_loader, criterion, device)
        test_loss_list.append(val_loss)
        accuracy_list.append(accuracy)

        # Store predictions and true labels for later analysis
        all_predictions.append(epoch_predictions)
        all_true_labels.append(epoch_true_labels)

        # Optionally adjust learning rate based on validation loss (if scheduler is used)
        if scheduler:
            scheduler.step(val_loss)

        # Print metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%")

    return train_loss_list, test_loss_list, accuracy_list, all_predictions, all_true_labels



# import torch
# import torch.nn as nn
# from typing import Optional, Tuple



# class PatchEmbedding(nn.Module):
#     """Patch Embedding layer for Vision Transformer.
    
#     This layer splits the input image into non-overlapping patches and projects each patch
#     into a vector of dimension embed_dim.
    
#     Args:
#         img_size (int): Size of input image (assumed square)
#         patch_size (int): Size of each patch (assumed square)
#         in_channels (int): Number of input channels
#         embed_dim (int): Dimension of the embedding space
#     """
#     def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
#         super().__init__()
#         self.num_patches = (img_size // patch_size) ** 2
#         self.projection = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the patch embedding layer.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size]
            
#         Returns:
#             torch.Tensor: Output tensor of shape [batch_size, num_patches, embed_dim]
#         """
#         x = self.projection(x)  # [batch_size, embed_dim, num_patches_h, num_patches_w]
#         x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
#         x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
#         return x

# class PositionEmbedding(nn.Module):
#     """Position Embedding layer for Vision Transformer.
    
#     Adds learnable position embeddings to the patch embeddings and prepends a learnable
#     class token.
    
#     Args:
#         num_patches (int): Number of patches
#         embed_dim (int): Dimension of the embedding space
#     """
#     def __init__(self, num_patches: int, embed_dim: int):
#         super().__init__()
#         self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the position embedding layer.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [batch_size, num_patches, embed_dim]
            
#         Returns:
#             torch.Tensor: Output tensor of shape [batch_size, num_patches + 1, embed_dim]
#         """
#         batch_size = x.size(0)
#         cls_token = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#         x = x + self.position_embedding
#         return x

# class TransformerEncoderBlock(nn.Module):
#     """Transformer Encoder Block for Vision Transformer.
    
#     Implements a single transformer encoder block with multi-head self-attention
#     and feed-forward network.
    
#     Args:
#         embed_dim (int): Dimension of the embedding space
#         num_heads (int): Number of attention heads
#         mlp_dim (int): Dimension of the feed-forward network
#         dropout (float): Dropout probability
#     """
#     def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
#         super().__init__()
#         self.layernorm1 = nn.LayerNorm(embed_dim)
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.layernorm2 = nn.LayerNorm(embed_dim)
        
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Linear(mlp_dim, embed_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the transformer encoder block.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
            
#         Returns:
#             torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
#         """
#         x = x + self.attention(self.layernorm1(x), self.layernorm1(x), self.layernorm1(x))[0]
#         x = x + self.mlp(self.layernorm2(x))
#         return x

# class VisionTransformer(nn.Module):
#     """Vision Transformer model for image classification.
    
#     Implements the Vision Transformer architecture as described in "An Image is Worth 16x16 Words:
#     Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020).
    
#     Args:
#         img_size (int): Size of input image (assumed square)
#         patch_size (int): Size of each patch (assumed square)
#         in_channels (int): Number of input channels
#         num_classes (int): Number of output classes
#         embed_dim (int): Dimension of the embedding space
#         num_heads (int): Number of attention heads
#         mlp_dim (int): Dimension of the feed-forward network
#         num_layers (int): Number of transformer encoder blocks
#         dropout (float): Dropout probability
#     """
#     def __init__(
#         self,
#         img_size: int = 224,
#         patch_size: int = 16,
#         in_channels: int = 3,
#         num_classes: int = 4,
#         embed_dim: int = 768,
#         num_heads: int = 8,
#         mlp_dim: int = 2048,
#         num_layers: int = 6,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         self.num_patches = (img_size // patch_size) ** 2
#         self.position_embedding = PositionEmbedding(self.num_patches, embed_dim)
        
#         self.encoder = nn.ModuleList([
#             TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
#             for _ in range(num_layers)
#         ])
        
#         self.layernorm = nn.LayerNorm(embed_dim)
#         self.classifier = nn.Linear(embed_dim, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass of the Vision Transformer.
        
#         Args:
#             x (torch.Tensor): Input tensor of shape [batch_size, in_channels, img_size, img_size]
            
#         Returns:
#             torch.Tensor: Output tensor of shape [batch_size, num_classes]
#         """
#         x = self.patch_embedding(x)
#         x = self.position_embedding(x)
        
#         for block in self.encoder:
#             x = block(x)
            
#         x = self.layernorm(x)
#         cls_token = x[:, 0]  # Extract class token
#         return self.classifier(cls_token) 