"""Evaluation metadata and fixed class ordering, independent of plotting and the GUI."""

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from sklearn.metrics import classification_report, confusion_matrix

from src.contracts import LABELS


def build_evaluation_report(
    y_true: Sequence[int], y_pred: Sequence[int], checkpoint_sha256: str, config: Mapping
) -> dict:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Evaluation requires non-empty, matching true and predicted labels.")
    if any(label not in (0, 1) for label in [*y_true, *y_pred]):
        raise ValueError("Evaluation labels must be NORMAL=0 or PNEUMONIA=1.")
    counts = Counter(y_true)
    # Supply both labels explicitly: a split with one class must still produce the
    # same two-row report/matrix, in checkpoint order rather than observed-label order.
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": checkpoint_sha256,
        "class_order": list(LABELS),
        "split": "test",
        "decision_rule": "argmax of two class logits",
        "configuration": {
            "image_size": config.get("image_size", 224),
            "batch_size": config.get("batch_size", 32),
            "normalisation_mean": 0.5,
            "normalisation_std": 0.5,
        },
        "dataset_counts": {label: counts[index] for index, label in enumerate(LABELS)},
        "sample_count": len(y_true),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
