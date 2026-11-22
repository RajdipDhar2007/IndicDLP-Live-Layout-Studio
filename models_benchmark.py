import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------------------------------------------------------------------
# 1. Classical CNN Architecture (LeNet-5 Adapted for 224x224 RGB Document Images)
# Canonical Udacity sequential feature extractor & classifier head
# ----------------------------------------------------------------------------------------
class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        
        # Convolutional feature extractor (conv -> bn -> relu -> pool)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            # Block 2
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            # Block 3
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28
        )
        
        self.gap = nn.AdaptiveAvgPool2d((7, 7))
        
        # Udacity Fully-connected classification head with Dropout
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Alias for backward compatibility
ModifiedLeNet5 = LeNet5


# ----------------------------------------------------------------------------------------
# 2. Transfer Learning Architecture (ResNet-18)
# Canonical Udacity Transfer Learning template: Pretrained backbone, freeze option,
# custom classifier head matching best_resnet18.pt
# ----------------------------------------------------------------------------------------
def build_resnet18(num_classes=2, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )
    return model
