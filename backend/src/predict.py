import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from src.model import load_trained_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "final_model.pth"

# Same preprocessing as agents/image_agent.py for FastAPI compatibility.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = INFERENCE_TRANSFORM(image.convert("RGB"))
    return tensor.unsqueeze(0)


def predict_from_tensor(tensor: torch.Tensor, model, device=None) -> dict:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        outputs = model(tensor.to(device))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    ranked = sorted(
        zip(CLASS_NAMES, probs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "diagnosis": CLASS_NAMES[top_idx],
        "confidence": round(probs[top_idx] * 100, 2),
        "ranked_predictions": [
            {"disease": disease, "confidence": round(prob * 100, 2)}
            for disease, prob in ranked
        ],
        "probabilities": {
            disease: round(prob * 100, 2) for disease, prob in zip(CLASS_NAMES, probs)
        },
    }


def predict_image(
    image_input: str | Path | bytes | Image.Image,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | None = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_trained_model(str(model_path), device=device)

    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise TypeError("image_input must be a path, bytes, or PIL Image")

    tensor = preprocess_image(image)
    return predict_from_tensor(tensor, model, device=device)


def predict_from_file_bytes(file_bytes: bytes, model, device=None) -> dict:
    """FastAPI-friendly helper: accepts raw bytes from image_agent pipeline."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tensor = preprocess_image(image)
    return predict_from_tensor(tensor, model, device=device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict diagnosis for a chest X-ray")
    parser.add_argument("image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    result = predict_image(args.image, model_path=args.model)
    print(f"Top diagnosis: {result['diagnosis']} ({result['confidence']}%)")
    print("\nAll predictions:")
    for item in result["ranked_predictions"]:
        print(f"  {item['disease']}: {item['confidence']}%")
