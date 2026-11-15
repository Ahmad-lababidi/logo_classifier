import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.models import DenseLogoCNN, resnet_model
from src.preprocessing import create_dataloaders
from src.utils import logger, load_checkpoint, save_checkpoint, save_class_names


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, model_kind: str):
    if model_kind == "resnet":
        backbone = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("fc.")]
        head = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith("fc.")]
        return optim.AdamW(
            [{"params": backbone, "lr": 1e-4}, {"params": head, "lr": 5e-4}],
            weight_decay=1e-4,
        )
    return optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    all_labels, all_predictions = [], []
    with torch.inference_mode():
        for images, labels in data_loader:
            logits = model(images.to(device))
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels, all_predictions, labels=list(range(num_classes)),
        average="macro", zero_division=0,
    )
    return accuracy, macro_f1, all_labels, all_predictions


def train_one_model(model, train_loader, val_loader, epochs, device, save_path, class_names, model_kind):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, model_kind)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    best_macro_f1 = -1.0
    best_accuracy = 0.0
    epochs_without_improvement = 0
    patience = 8

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)

        val_accuracy, val_macro_f1, _, _ = evaluate(
            model, val_loader, device, num_classes=len(class_names)
        )
        scheduler.step(val_macro_f1)
        logger.info(
            "%s epoch %d/%d | loss %.4f | val accuracy %.4f | val macro-F1 %.4f",
            model_kind, epoch, epochs, running_loss / max(seen, 1), val_accuracy, val_macro_f1,
        )

        if (val_macro_f1, val_accuracy) > (best_macro_f1, best_accuracy):
            best_macro_f1, best_accuracy = val_macro_f1, val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                save_path,
                metadata={
                    "model": model_kind,
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                    "validation_macro_f1": val_macro_f1,
                    "class_count": len(class_names),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("%s early stopping after epoch %d.", model_kind, epoch)
                break

    model = load_checkpoint(model, save_path, device)
    accuracy, macro_f1, labels, predictions = evaluate(
        model, val_loader, device, num_classes=len(class_names)
    )
    report = classification_report(
        labels, predictions, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    logger.info("%s best validation accuracy %.4f | macro-F1 %.4f", model_kind, accuracy, macro_f1)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": report}


def fit_knn(model, train_loader, val_loader, device, class_names, save_path):
    model.to(device)
    model.eval()
    train_features, train_labels = [], []
    val_features, val_labels = [], []

    with torch.inference_mode():
        for images, labels in train_loader:
            train_features.append(model.extract_features(images.to(device)).cpu().numpy())
            train_labels.append(labels.numpy())
        for images, labels in val_loader:
            val_features.append(model.extract_features(images.to(device)).cpu().numpy())
            val_labels.append(labels.numpy())

    x_train = np.concatenate(train_features)
    y_train = np.concatenate(train_labels)
    x_val = np.concatenate(val_features)
    y_val = np.concatenate(val_labels)
    neighbors = min(5, len(x_train))
    classifier = KNeighborsClassifier(
        n_neighbors=neighbors,
        weights="distance",
        metric="cosine",
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, labels=list(range(len(class_names))),
        average="macro", zero_division=0,
    )
    report = classification_report(
        y_val, predictions, labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    joblib.dump(classifier, save_path)
    logger.info(
        "CNN + kNN validation accuracy %.4f | macro-F1 %.4f (k=%d)",
        accuracy, macro_f1, neighbors,
    )
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": report}


def main():
    parser = argparse.ArgumentParser(description="Train and compare logo classifiers.")
    parser.add_argument("--data_dir", type=str, default="data/1000 Logos - Temp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)
    train_loader, val_loader, train_eval_loader, class_names = create_dataloaders(
        args.data_dir, batch_size=args.batch_size, val_split=args.val_split, seed=args.seed
    )
    save_class_names(class_names, "saved_models/class_names.json")
    logger.info("Training two models for %d classes.", len(class_names))

    scratch_metrics = train_one_model(
        DenseLogoCNN(num_classes=len(class_names)), train_loader, val_loader,
        epochs=args.epochs, device=device, save_path="saved_models/scratch_cnn.pth",
        class_names=class_names, model_kind="scratch_cnn",
    )
    scratch_model = load_checkpoint(
        DenseLogoCNN(num_classes=len(class_names)),
        "saved_models/scratch_cnn.pth",
        device,
    )
    knn_metrics = fit_knn(
        scratch_model,
        train_eval_loader,
        val_loader,
        device,
        class_names,
        "saved_models/scratch_knn.joblib",
    )
    resnet_metrics = train_one_model(
        resnet_model(num_classes=len(class_names), pretrained=True), train_loader, val_loader,
        epochs=args.epochs, device=device, save_path="saved_models/resnet_finetuned.pth",
        class_names=class_names, model_kind="resnet",
    )

    metrics = {
        "seed": args.seed,
        "validation_split": args.val_split,
        "class_count": len(class_names),
        "training_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "unique_images": len(train_loader.dataset) + len(val_loader.dataset),
        "image_size": [224, 224],
        "scratch_cnn": scratch_metrics,
        "scratch_knn": knn_metrics,
        "resnet": resnet_metrics,
    }
    metrics_path = Path("saved_models/training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved validation metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
