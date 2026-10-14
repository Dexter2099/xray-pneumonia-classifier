"""Train the existing CNN using explicit, validated and seeded configuration."""

import random
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.errors import ConfigurationError
from src.model import SimpleCNN
from src.settings import TrainingSettings


def build_optimizer(model: nn.Module, settings: TrainingSettings) -> optim.Optimizer:
    parameters = {
        "lr": settings.learning_rate,
        "weight_decay": settings.weight_decay,
    }
    if settings.optimizer == "adam":
        return optim.Adam(model.parameters(), **parameters)
    if settings.optimizer == "sgd":
        return optim.SGD(model.parameters(), momentum=0.9, **parameters)
    raise ConfigurationError("Optimizer must be adam or sgd.")


def train_model(config: Mapping) -> None:
    settings = TrainingSettings.from_mapping(config)
    # Seed initialization, dropout and sampling together. This improves repeatability
    # within a runtime; it does not promise bitwise equality across GPUs/versions.
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = settings.inference.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA was requested but is unavailable. Select cpu or auto.")
    device = torch.device(device_name if requested_device == "auto" else requested_device)

    # Model selection uses validation only; training must not depend on the held-out test set.
    train_loader, val_loader = get_dataloaders(config, splits=("train", "val"), generator=generator)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, settings)
    checkpoint_path = settings.inference.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(settings.epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{settings.epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            # CrossEntropyLoss expects logits and applies its own log-softmax.
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_acc = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
    print("Training complete.")


def evaluate(model, dataloader, device) -> float:
    """Validation accuracy for model selection; final test reporting lives in evaluate.py."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise ConfigurationError("Cannot calculate accuracy for an empty dataset.")
    return correct / total
