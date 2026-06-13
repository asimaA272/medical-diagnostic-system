import torch

from src.dataset import CLASS_NAMES
from src.model import load_trained_model

DISEASES = CLASS_NAMES


def load_model(model_path: str):
    return load_trained_model(model_path, device="cpu")

def detection_agent(tensor, model):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs.cpu().numpy()[0]
