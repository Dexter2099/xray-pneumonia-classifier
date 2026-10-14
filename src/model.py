"""Original checkpoint architecture; layer names and dimensions are compatibility constraints."""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Return two class logits from one 224 x 224 grayscale image per batch item."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Three 2x pooling steps reduce 224 to 28. Changing this dense layer would
        # change checkpoint parameter shapes, so incompatible input sizes are rejected.
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)  # model.eval() disables this during prediction/Grad-CAM.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 56, 56)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 28, 28)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # Training's CrossEntropyLoss consumes logits; inference applies softmax once
        # for display. Class selection remains argmax, without a separate threshold.
        return x
