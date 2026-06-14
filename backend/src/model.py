import torch
import torch.nn as nn
from torchvision import models
from src.dataset import CLASS_NAMES, NUM_CLASSES

def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.5, pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model

def load_trained_model(
    model_path: str,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.5,
    device: str = "cpu",
):
    model = build_model(num_classes=num_classes, dropout=dropout, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_class_names():
    return CLASS_NAMES.copy()