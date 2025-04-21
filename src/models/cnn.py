import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """CNN model for brain tumor classification.
    
    Architecture:
        - 4 convolutional layers with LeakyReLU activation and max pooling
        - 2 fully connected layers
        - Input shape: (batch_size, 1, 128, 128)
        - Output shape: (batch_size, 4) for 4 tumor classes
    """
    def __init__(self):
        super(CNNModel, self).__init__()

        # Conv 1: 128x128 -> 124x124 -> 62x62
        self.cnv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Conv 2: 62x62 -> 58x58 -> 29x29
        self.cnv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Conv 3: 29x29 -> 25x25 -> 12x12
        self.cnv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Conv 4: 12x12 -> 8x8 -> 4x4
        self.cnv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Activation Function
        self.leakyRelu = nn.LeakyReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 4)  # 4 classes for tumor classification

    def forward(self, x):
        # Layer 1
        out = self.leakyRelu(self.cnv1(x))
        out = self.maxpool1(out)

        # Layer 2
        out = self.leakyRelu(self.cnv2(out))
        out = self.maxpool2(out)

        # Layer 3
        out = self.leakyRelu(self.cnv3(out))
        out = self.maxpool3(out)

        # Layer 4
        out = self.leakyRelu(self.cnv4(out))
        out = self.maxpool4(out)

        # Flatten
        out = out.view(out.size(0), -1)

        # Linear layers
        out = self.leakyRelu(self.fc1(out))
        out = self.fc2(out)

        return out
    

class BrainTumorCNN(nn.Module):
    """
    CNN architecture for brain tumor classification.

    Architecture:
    - 4 convolutional blocks (each with Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d)
    - 2 fully connected layers with dropout
    """
    def __init__(self, num_classes: int = 4):
        super(BrainTumorCNN, self).__init__()

        # Convolutional layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            # Third conv block
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),  # Add dropout to prevent overfitting
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = torch.tanh(self.bn1(self.conv1(x)))  # Use Tanh activation
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.tanh(out)  # Use Tanh activation
        return out
    
# Define the BrainTumorCNN with ResNet blocks
class BrainTumorCNN_RN(nn.Module):
    """
    CNN architecture for brain tumor classification with ResNet blocks.

    Architecture:
    - 2 convolutional blocks
    - 2 residual blocks
    - 2 fully connected layers with dropout
    """
    def __init__(self, num_classes: int = 4):
        super(BrainTumorCNN_RN, self).__init__()

        # Convolutional and residual layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            # Second conv block
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            # First residual block
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),

            # Second residual block
            ResidualBlock(64, 128),
            nn.MaxPool2d(2)
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 1024),  # Adjusted input size
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x