import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseLogoCNN(nn.Module):

    def __init__(self, num_classes: int, dropout_rate: float = 0.4):
        super(DenseLogoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )
        self.fc = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        emb = self.embedding(x)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        emb = self.embedding(x)
        return self.fc(emb)

def resnet_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if pretrained and freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model