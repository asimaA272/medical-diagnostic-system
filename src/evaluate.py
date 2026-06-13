import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.dataset import CLASS_NAMES, ChestXrayDataset, collect_samples, get_val_transforms
from src.model import load_trained_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "final_model.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "model" / "confusion_matrix.png"


def evaluate_model(model, dataset, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for image, label in tqdm(dataset, desc="Evaluating"):
            image = image.unsqueeze(0).to(device)
            outputs = model(image)
            pred = outputs.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate chest X-ray classifier")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.model, device=device)

    samples = collect_samples(args.data_dir)
    dataset = ChestXrayDataset(samples, transform=get_val_transforms())

    y_true, y_pred = evaluate_model(model, dataset, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 47)
    for idx, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<15} "
            f"{precision[idx] * 100:>9.2f}% "
            f"{recall[idx] * 100:>9.2f}% "
            f"{f1[idx] * 100:>9.2f}%"
        )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, CLASS_NAMES, Path(args.output))
    print(f"\nConfusion matrix plot saved to {args.output}")


if __name__ == "__main__":
    main()
