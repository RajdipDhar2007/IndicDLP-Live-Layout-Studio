# ========================================================================================
# IndicDLP: PyTorch Dataset with Aspect-Ratio Preserving Letterboxing Transform
# Handles variable bounding boxes without distortion (Pads to 224x224)
# ========================================================================================

import os
from pathlib import Path
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class AspectRatioPad:
    """
    Scales the longest edge of an arbitrary bounding box crop to target_size (224x224)
    while maintaining its exact aspect ratio, then centers it on a neutral broadsheet
    canvas (fill=255) to prevent character stroke distortion.
    """
    def __init__(self, target_size=(224, 224), fill=255):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Scale preserving aspect ratio
        img = img.convert("RGB")
        img.thumbnail(self.target_size, Image.Resampling.BICUBIC)
        
        # 2. Symmetric padding to target size
        delta_w = self.target_size[0] - img.size[0]
        delta_h = self.target_size[1] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding, fill=self.fill)

def get_transforms(split="train"):
    """Returns PyTorch transform pipeline for training or evaluation."""
    base_transforms = [
        AspectRatioPad(target_size=(224, 224), fill=255),
    ]

    if split == "train":
        # Data augmentations realistic for historical letterpress print
        augmentations = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-3, 3), fill=255),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
        base_transforms.extend(augmentations)

    base_transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(base_transforms)

class HistoricalAdDataset(Dataset):
    """
    PyTorch Dataset loading from Desktop/Ad_Dataset partitioned directories.
    Classes: 0 = non_ad, 1 = ad
    """
    def __init__(self, root_dir, split="train"):
        self.split_dir = Path(root_dir) / split
        self.transform = get_transforms(split)
        self.samples = []

        ad_dir = self.split_dir / "ad"
        non_ad_dir = self.split_dir / "non_ad"

        if ad_dir.exists():
            for p in ad_dir.glob("*.*"):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((str(p), 1))

        if non_ad_dir.exists():
            for p in non_ad_dir.glob("*.*"):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((str(p), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            tensor = self.transform(img)
        return tensor, label
