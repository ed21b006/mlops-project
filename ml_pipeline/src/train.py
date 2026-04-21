# DVC Stage 3: Model Training (Memory Optimized)

import gc
import itertools
import json
import logging
import os
import sys
import time
import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MNISTNet
from mlflow_utils import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_one_epoch(model, train_loader, optimizer):
    # Train for one epoch, return average loss and accuracy
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        if imgs.dtype != torch.float32:
            imgs = imgs.float()
        if imgs.max().item() > 1.0:
            imgs = imgs / 255.0

        optimizer.zero_grad()
        out = model(imgs)
        loss = F.nll_loss(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += len(labels)

    return total_loss / len(train_loader), correct / total


def validate(model, val_loader):
    # Validate model, return loss, accuracy, and F1 score
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            if imgs.dtype != torch.float32:
                imgs = imgs.float()
            if imgs.max().item() > 1.0:
                imgs = imgs / 255.0

            out = model(imgs)
            val_loss += F.nll_loss(out, labels).item()
            preds = out.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return (
        val_loss / len(val_loader),
        correct / total,
        f1_score(all_labels, all_preds, average="macro"),
    )


def build_search_grid(params):
    # Build hyperparameter combinations from params.yaml
    keys, values = [], []
    for category in ["train", "model"]:
        for k, v in params[category].items():
            keys.append(k)
            values.append(v if isinstance(v, list) else [v])

    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def main():
    # Assisted by AI
    # Tune thread usage for constrained containers.
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))

    # Load parameters
    with open(os.path.join(BASE_DIR, "params.yaml")) as f:
        params = yaml.safe_load(f)

    combinations = build_search_grid(params)
    logger.info(f"Hyperparameter search: {len(combinations)} combinations")

    logger.info("Loading training data...")
    train_data = torch.load(os.path.join(BASE_DIR, "data", "transformed", "train.pt"), weights_only=True, map_location="cpu")
    train_images = train_data["images"]
    train_labels = train_data["labels"]
    del train_data
    gc.collect()

    logger.info("Loading validation data...")
    val_data = torch.load(os.path.join(BASE_DIR, "data", "transformed", "val.pt"), weights_only=True, map_location="cpu")
    val_images = val_data["images"]
    val_labels = val_data["labels"]
    del val_data
    gc.collect()


    logger.info(f"Training on {len(train_images)} samples, validating on {len(val_images)} samples")

    # MLflow setup
    setup_mlflow()

    best_f1 = 0.0
    best_run_id = None
    best_model_state = None
    best_arch_params = None
    best_params = None
    all_results = []

    with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
        logger.info(f"Parent run: {parent_run.info.run_id}")

        for i, hp in enumerate(combinations):
            run_name = f"exp_{i + 1}"

            with mlflow.start_run(nested=True, run_name=run_name) as child_run:
                mlflow.log_params(hp)

                # Create DataLoaders with num_workers=0 to avoid forking memory overhead
                train_loader = DataLoader(
                    TensorDataset(train_images, train_labels),
                    batch_size=hp["batch_size"], shuffle=True, num_workers=0,
                )
                val_loader = DataLoader(
                    TensorDataset(val_images, val_labels),
                    batch_size=hp["batch_size"], shuffle=False, num_workers=0,
                )

                model = MNISTNet(
                    conv1_ch=hp["conv1_channels"],
                    conv2_ch=hp["conv2_channels"],
                    fc1_units=hp["fc1_units"],
                    dropout=hp["dropout"],
                )

                if hp["optimizer"] == "adam":
                    opt = optim.Adam(model.parameters(), lr=hp["learning_rate"])
                elif hp["optimizer"] == "sgd":
                    opt = optim.SGD(model.parameters(), lr=hp["learning_rate"])
                else:
                    raise ValueError(f"Unknown optimizer: {hp['optimizer']}")

                start_time = time.time()

                # Training loop
                for epoch in range(1, hp["epochs"] + 1):
                    train_loss, train_acc = train_one_epoch(model, train_loader, opt)
                    val_loss, val_acc, val_f1 = validate(model, val_loader)

                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                    }, step=epoch)

                    logger.info(
                        f"[{run_name}] Epoch {epoch}/{hp['epochs']} — "
                        f"train_loss={train_loss:.4f}, val_f1={val_f1:.4f}"
                    )

                elapsed = time.time() - start_time
                mlflow.log_metric("training_time_sec", elapsed)
                mlflow.log_metric("final_val_f1", val_f1)

                all_results.append({
                    "run": run_name,
                    "run_id": child_run.info.run_id,
                    "params": hp,
                    "val_f1": round(val_f1, 6),
                    "val_acc": round(val_acc, 6),
                    "training_time_sec": round(elapsed, 2),
                })

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_run_id = child_run.info.run_id
                    # Deep copy state dict to avoid holding reference to full model
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    best_arch_params = model.arch_params
                    best_params = hp

                # Explicitly free model + optimizer between experiments
                del model, opt, train_loader, val_loader
                gc.collect()

        logger.info(f"Best run: {best_run_id} with F1={best_f1:.4f}")

        # Tag best run
        client = MlflowClient()
        client.set_tag(best_run_id, "best_model", "true")

    # Free training data before saving
    del train_images, train_labels, val_images, val_labels
    gc.collect()

    # Save best model checkpoint
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "models", "mnist_cnn.pth")
    torch.save({
        "model_state_dict": best_model_state,
        "arch_params": best_arch_params,
    }, model_path)
    logger.info(f"Best model saved to {model_path}")

    del best_model_state
    gc.collect()

    # Save training results for DVC metrics
    os.makedirs(os.path.join(BASE_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "metrics", "train_results.json"), "w") as f:
        json.dump({
            "best_run_id": best_run_id,
            "best_val_f1": round(best_f1, 6),
            "best_params": best_params,
            "total_experiments": len(combinations),
            "all_results": all_results,
        }, f, indent=2)

    logger.info("Training stage complete.")


if __name__ == "__main__":
    main()
