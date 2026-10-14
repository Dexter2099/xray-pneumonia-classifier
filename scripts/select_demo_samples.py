"""Select presentation examples from real outputs without implementing another predictor."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAMPLE_DIR = ROOT / "demo-assets" / "sample-xrays"
METADATA_PATH = SAMPLE_DIR / "demo_samples.json"
DATASET_DIRS = [
    ROOT / "data" / "raw" / "chest_xray" / "test" / "NORMAL",
    ROOT / "data" / "raw" / "chest_xray" / "test" / "PNEUMONIA",
    ROOT / "data" / "raw" / "chest_xray" / "val" / "NORMAL",
    ROOT / "data" / "raw" / "chest_xray" / "val" / "PNEUMONIA",
    ROOT / "data" / "raw" / "chest_xray" / "train" / "NORMAL",
    ROOT / "data" / "raw" / "chest_xray" / "train" / "PNEUMONIA",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OUTPUT_FILENAMES = {
    "high_confidence_pneumonia": "pneumonia-high-confidence.jpeg",
    "normal": "normal-example.jpeg",
    "borderline": "borderline-pneumonia.jpeg",
}
CASE_TITLES = {
    "high_confidence_pneumonia": "High-confidence pneumonia",
    "normal": "Normal X-ray",
    "borderline": "Borderline case",
}


def source_has_class_folder(source_path: str, class_name: str) -> bool:
    return any(part.upper() == class_name for part in Path(source_path).parts)


def discover_images(dataset_dirs: list[Path] = DATASET_DIRS) -> list[Path]:
    images: list[Path] = []
    for directory in dataset_dirs:
        if not directory.exists():
            continue
        images.extend(
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return images


def score_gradcam_quality(cam) -> float:
    """Presentation heuristic for demo selection, not a measure of clinical validity."""
    cam_array = np.asarray(cam, dtype=np.float32)
    if cam_array.size == 0:
        return 0.0
    cam_array = cam_array - float(np.min(cam_array))
    max_value = float(np.max(cam_array))
    if max_value <= 1e-8:
        return 0.0
    cam_array = cam_array / max_value
    contrast = float(np.percentile(cam_array, 95) - np.percentile(cam_array, 5))
    high_activation_area = float(np.mean(cam_array >= 0.55))
    variance = float(np.var(cam_array))
    return round((contrast * 0.45) + (high_activation_area * 0.35) + (variance * 0.20), 6)


def compute_gradcam_score(model, input_tensor) -> float:
    from src.gradcam import GradCAM

    gradcam = GradCAM(model, model.conv3)
    cam = gradcam.generate(input_tensor)
    return score_gradcam_quality(cam)


def predict_candidates(image_paths: list[Path], config: dict) -> list[dict]:
    from src.inference import PredictionService
    from src.inputs import load_image, preprocess_image
    from src.settings import InferenceSettings

    # Selection is another consumer of inference, not a second implementation of it.
    service = PredictionService(InferenceSettings.from_mapping(config))

    candidates = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        try:
            image = load_image(image_path)
            result = service.predict(image).to_dict()
            gradcam_score = None
            if result["label"] == "PNEUMONIA":
                tensor = preprocess_image(image).unsqueeze(0).to(service.device)
                with service.lock:
                    gradcam_score = compute_gradcam_score(service.model, tensor)
        except Exception as exc:
            print(f"[WARN] Skipping {image_path}: {exc}", file=sys.stderr)
            continue

        candidates.append(
            {
                "source_path": str(image_path),
                "predicted_label": result["label"],
                "confidence": float(result["confidence"]),
                "probabilities": {
                    "NORMAL": float(result["probabilities"]["NORMAL"]),
                    "PNEUMONIA": float(result["probabilities"]["PNEUMONIA"]),
                },
                "gradcam_score": gradcam_score,
            }
        )
        if index % 100 == 0:
            print(f"[INFO] Scored {index}/{total} images...")
    return candidates


def select_best_samples(candidates: list[dict]) -> dict[str, dict]:
    pneumonia_candidates = [item for item in candidates if item["predicted_label"] == "PNEUMONIA"]
    normal_candidates = [item for item in candidates if item["predicted_label"] == "NORMAL"]
    if not pneumonia_candidates:
        raise RuntimeError("No model-predicted PNEUMONIA images found.")
    if not normal_candidates:
        raise RuntimeError("No model-predicted NORMAL images found.")

    labelled_pneumonia_candidates = [
        item
        for item in pneumonia_candidates
        if source_has_class_folder(item["source_path"], "PNEUMONIA")
    ]
    high_confidence_pool = labelled_pneumonia_candidates or pneumonia_candidates

    def pneumonia_selection_score(item):
        probability = item["probabilities"]["PNEUMONIA"]
        gradcam_score = float(item.get("gradcam_score") or 0.0)
        in_target_range = 0.85 <= probability <= 0.95
        in_acceptable_range = 0.80 <= probability <= 0.98
        distance_from_target = abs(probability - 0.90)
        overconfident_penalty = max(probability - 0.98, 0.0) * 6.0
        return (
            2.0 if in_target_range else 0.0,
            1.0 if in_acceptable_range else 0.0,
            gradcam_score,
            -distance_from_target - overconfident_penalty,
        )

    high_confidence_pneumonia = max(
        high_confidence_pool,
        key=pneumonia_selection_score,
    )
    normal = max(
        normal_candidates,
        key=lambda item: (
            item["confidence"] >= 0.80,
            item["confidence"],
        ),
    )
    borderline = min(
        pneumonia_candidates,
        key=lambda item: abs(item["probabilities"]["PNEUMONIA"] - 0.50),
    )
    return {
        "high_confidence_pneumonia": high_confidence_pneumonia,
        "normal": normal,
        "borderline": borderline,
    }


def build_metadata_entry(case_id: str, candidate: dict, filename: str) -> dict:
    pneumonia_probability = candidate["probabilities"]["PNEUMONIA"]
    descriptions = {
        "high_confidence_pneumonia": "Real model output with high pneumonia probability.",
        "normal": "Real model output for a normal / no-pneumonia case.",
        "borderline": (
            "Near the model decision boundary "
            f"with {pneumonia_probability:.1%} pneumonia probability."
        ),
    }
    entry = {
        "case_id": case_id,
        "filename": filename,
        "source_path": candidate["source_path"],
        "predicted_label": candidate["predicted_label"],
        "confidence": round(float(candidate["confidence"]), 6),
        "normal_probability": round(float(candidate["probabilities"]["NORMAL"]), 6),
        "pneumonia_probability": round(float(pneumonia_probability), 6),
        "case_title": CASE_TITLES[case_id],
        "case_description": descriptions[case_id],
    }
    if candidate.get("gradcam_score") is not None:
        entry["gradcam_quality_score"] = round(float(candidate["gradcam_score"]), 6)
    return entry


def write_selected_samples(selected: dict[str, dict], sample_dir: Path = SAMPLE_DIR) -> list[dict]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for case_id, candidate in selected.items():
        filename = OUTPUT_FILENAMES[case_id]
        destination = sample_dir / filename
        shutil.copyfile(candidate["source_path"], destination)
        metadata.append(build_metadata_entry(case_id, candidate, filename))

    (sample_dir / "demo_samples.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_config() -> dict:
    from src.settings import load_config as read_config

    return read_config()


def main() -> int:
    from src.contracts import LABELS
    from src.settings import InferenceSettings, resolve_project_path

    config = load_config()
    checkpoint_path = InferenceSettings.from_mapping(config).checkpoint_path
    if not checkpoint_path.exists():
        print(f"Missing trained checkpoint: {checkpoint_path}", file=sys.stderr)
        return 1

    data_dir = resolve_project_path(config["data_dir"], "data_dir")
    image_paths = discover_images(
        [data_dir / split / label for split in ("test", "val", "train") for label in LABELS]
    )
    if not image_paths:
        print(f"No local X-ray images found under {data_dir}.", file=sys.stderr)
        return 1

    print(f"[INFO] Scoring {len(image_paths)} local X-ray images with the real checkpoint...")
    candidates = predict_candidates(image_paths, config)
    selected = select_best_samples(candidates)
    metadata = write_selected_samples(selected)

    print(f"[INFO] Wrote {METADATA_PATH}")
    for item in metadata:
        print(
            "[INFO] "
            f"{item['case_title']}: {item['predicted_label']} "
            f"confidence={item['confidence']:.1%}, "
            f"pneumonia={item['pneumonia_probability']:.1%}, "
            f"file={item['filename']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
