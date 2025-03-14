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