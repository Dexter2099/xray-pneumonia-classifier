"""Evaluate the fixed checkpoint on the test split, without loading training/validation data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import ChestXrayDataset
from src.inference import LABELS, PredictionService
from src.reporting import build_evaluation_report
from src.settings import ROOT, InferenceSettings


def evaluate_model(config):
    settings = InferenceSettings.from_mapping(config)
    data_dir = (ROOT / Path(config["data_dir"])).resolve()
    batch_size = config.get("batch_size", 32)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Evaluation batch_size must be a positive integer.")
    dataset = ChestXrayDataset(data_dir, "test", image_size=224)
    records = sorted(zip(dataset.image_paths, dataset.labels, strict=True))
    if not records:
        raise ValueError("The test dataset is empty.")
    service = PredictionService(settings)
    y_true, y_pred = [], []
    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        predictions = service.predict_batch([Path(path) for path, _ in batch])
        y_true.extend(label for _, label in batch)
        y_pred.extend(LABELS.index(prediction.label) for prediction in predictions)
    report = build_evaluation_report(y_true, y_pred, service.checkpoint_sha256, config)
    save_dir = (ROOT / Path(config["visualizations_dir"])).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_confusion_matrix(report["confusion_matrix"], save_dir)
    print(
        f"Evaluated {report['sample_count']} test images; report saved to {save_dir / 'metrics.json'}"
    )
    return report


def plot_confusion_matrix(cm, save_dir):
    figure, axis = plt.subplots(figsize=(6, 5))
    try:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axis,
            xticklabels=list(LABELS),
            yticklabels=list(LABELS),
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Confusion Matrix")
        figure.tight_layout()
        figure.savefig(Path(save_dir) / "confusion_matrix.png")
    finally:
        plt.close(figure)
