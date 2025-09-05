import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch

# Get the absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "assets", "emotion_cnn.pth")

if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location="cpu")
else:
    raise FileNotFoundError(f"❗ Model file not found at {MODEL_PATH}")


class EmotionCNN(nn.Module):
    """
    A simple Convolutional Neural Network for facial emotion recognition.

    The model consists of two convolutional layers followed by max pooling,
    a dropout layer for regularization, and two fully connected layers
    for classification.
    """

    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # First convolutional block with increased filters and batch normalization
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Second convolutional block with increased filters and batch normalization
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Third convolutional block
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.5)

        # Fully connected layers
        # The input size to the linear layer changes, you need to calculate this
        # For a 48x48 input image:
        # After pool1 (48/2=24): 128 channels, 24x24
        # After pool2 (24/2=12): 256 channels, 12x12
        # After pool3 (12/2=6): 512 channels, 6x6
        self.fc1 = nn.Linear(512 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)

