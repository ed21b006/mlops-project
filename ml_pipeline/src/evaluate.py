# DVC Stage 4: Model Evaluation

import json
import logging
import os
import sys

import mlflow
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MNISTNet
from mlflow_utils import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Load model checkpoint
    model_path = os.path.join(BASE_DIR, "models", "mnist_cnn.pth")
    checkpoint = torch.load(model_path, weights_only=True)
    model = MNISTNet(**checkpoint["arch_params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    test_data = torch.load(
        os.path.join(BASE_DIR, "data", "transformed", "test.pt"), weights_only=True
    )
    loader = DataLoader(
        TensorDataset(test_data["images"], test_data["labels"]),
        batch_size=256,
        shuffle=False,
    )

    # Run predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if imgs.dtype != torch.float32:
                imgs = imgs.float()
            if imgs.max().item() > 1.0:
                imgs = imgs / 255.0

            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"Test — accuracy={acc:.4f}, F1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}")

    # Log to MLflow
    setup_mlflow()
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "test_precision_macro": precision,
            "test_recall_macro": recall,
        })

    # Save metrics for DVC tracking
    os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "metrics", "eval_metrics.json"), "w") as f:
        json.dump({
            "accuracy": round(acc, 6),
            "f1_score_macro": round(f1, 6),
            "precision_macro": round(precision, 6),
            "recall_macro": round(recall, 6),
        }, f, indent=2)

    # Confusion matrix for DVC plots
    cm_data = [
        {"actual": str(a), "predicted": str(p), "count": int(cm[a][p])}
        for a in range(10) for p in range(10)
    ]
    with open(os.path.join(BASE_DIR, "metrics", "confusion_matrix.json"), "w") as f:
        json.dump(cm_data, f, indent=2)

    logger.info("Evaluation stage complete.")


if __name__ == "__main__":
    main()
