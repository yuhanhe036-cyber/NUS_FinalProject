"""CNN architecture and preprocessing shared by Expert Tasks 1 and 2."""

import cv2
import numpy as np
import torch
from torch import nn


INPUT_SIZE = 48
NORMALIZATION_MEAN = 0.5
NORMALIZATION_STD = 0.5


class ExpressionCNN(nn.Module):
    """Compact CNN designed for 48x48 grayscale facial-expression images."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def preprocess_gray_face(gray_face):
    """Convert one grayscale face crop into the exact CNN inference tensor."""
    if gray_face is None or gray_face.size == 0:
        return None
    resized = cv2.resize(gray_face, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - NORMALIZATION_MEAN) / NORMALIZATION_STD
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
