# Task 2.1: Custom Deep CNN Architectures (3, 4, and 5 layers)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3Layer(nn.Module):
    """
    Purpose: A 3-layer Convolutional Neural Network for binary classification.
    Inputs: num_classes (int) - The number of target classes (2).
    Outputs: Logits for each class.
    Assumptions: Input images are RGB and resized to 224x224.
    """
    def __init__(self, num_classes=2):
        super(CNN3Layer, self).__init__()
        # Block 1: Input (3, 224, 224) -> Conv -> Pool -> Output (32, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Input (32, 112, 112) -> Conv -> Pool -> Output (64, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Input (64, 56, 56) -> Conv -> Pool -> Output (128, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten: 128 channels * 28 height * 28 width = 100,352 neurons
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN4Layer(nn.Module):
    """
    Purpose: A 4-layer CNN adding deeper feature extraction.
    Inputs: num_classes (int) - Default 2.
    Outputs: Logits for each class.
    """
    def __init__(self, num_classes=2):
        super(CNN4Layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),    # -> 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # -> 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # -> 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)  # -> 14x14
        )
        # Flatten: 256 channels * 14 height * 14 width = 50,176 neurons
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN5Layer(nn.Module):
    """
    Purpose: A 5-layer CNN for maximum custom depth.
    Inputs: num_classes (int) - Default 2.
    Outputs: Logits for each class.
    """
    def __init__(self, num_classes=2):
        super(CNN5Layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),    # -> 112x112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # -> 56x56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # -> 28x28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 14x14
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)  # -> 7x7
        )
        # Flatten: 512 channels * 7 height * 7 width = 25,088 neurons
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- SANITY CHECK ---
# Let's push a dummy tensor (Batch Size=2, Channels=3, H=224, W=224) through each model
# to ensure the architecture connects properly before we waste time on a training loop.

print("Running architecture shape verification...")
dummy_input = torch.randn(2, 3, 224, 224)

models = {
    "3-Layer CNN": CNN3Layer(num_classes=2),
    "4-Layer CNN": CNN4Layer(num_classes=2),
    "5-Layer CNN": CNN5Layer(num_classes=2)
}

for name, model in models.items():
    try:
        output = model(dummy_input)
        print(f"[{name}] Success! Output shape: {output.shape} (Expected: [2, 2])")
    except Exception as e:
        print(f"[{name}] Error: {e}")