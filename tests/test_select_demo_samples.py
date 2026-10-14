from PIL import Image

from scripts import select_demo_samples
from src.inference import Prediction


def test_select_best_samples_prefers_high_confidence_pneumonia_with_useful_gradcam():
    candidates = [
        {
            "source_path": "normal-low.jpeg",
            "predicted_label": "NORMAL",
            "confidence": 0.82,
            "probabilities": {"NORMAL": 0.82, "PNEUMONIA": 0.18},
        },
        {
            "source_path": "C:/repo/xray-pneumonia-classifier/data/raw/chest_xray/test/PNEUMONIA/pneumonia-high.jpeg",
            "predicted_label": "PNEUMONIA",
            "confidence": 0.91,
            "probabilities": {"NORMAL": 0.09, "PNEUMONIA": 0.91},
            "gradcam_score": 0.65,
        },
        {
            "source_path": "C:/repo/xray-pneumonia-classifier/data/raw/chest_xray/test/PNEUMONIA/flat-100.jpeg",
            "predicted_label": "PNEUMONIA",
            "confidence": 1.0,
            "probabilities": {"NORMAL": 0.0, "PNEUMONIA": 1.0},
            "gradcam_score": 0.02,
        },
        {
            "source_path": "C:/repo/xray-pneumonia-classifier/data/raw/chest_xray/test/NORMAL/false-positive-higher.jpeg",
            "predicted_label": "PNEUMONIA",
            "confidence": 0.99,
            "probabilities": {"NORMAL": 0.01, "PNEUMONIA": 0.99},
            "gradcam_score": 0.99,
        },
        {
            "source_path": "pneumonia-borderline.jpeg",
            "predicted_label": "PNEUMONIA",
            "confidence": 0.55,
            "probabilities": {"NORMAL": 0.45, "PNEUMONIA": 0.55},
        },
        {
            "source_path": "pneumonia-middle.jpeg",
            "predicted_label": "PNEUMONIA",
            "confidence": 0.70,
            "probabilities": {"NORMAL": 0.30, "PNEUMONIA": 0.70},
        },
    ]

    selected = select_demo_samples.select_best_samples(candidates)

    assert selected["high_confidence_pneumonia"]["source_path"].endswith(
        "test/PNEUMONIA/pneumonia-high.jpeg"
    )
    assert selected["normal"]["source_path"] == "normal-low.jpeg"
    assert selected["borderline"]["source_path"] == "pneumonia-borderline.jpeg"


def test_score_gradcam_quality_prefers_contrast_and_activation_area():
    flat_score = select_demo_samples.score_gradcam_quality([[0.0, 0.0], [0.0, 0.0]])
    useful_score = select_demo_samples.score_gradcam_quality([[0.0, 0.2], [0.8, 1.0]])

    assert flat_score == 0.0
    assert useful_score > flat_score


def test_high_confidence_metadata_avoids_clinical_certainty_language():
    candidate = {
        "source_path": "source.jpeg",
        "predicted_label": "PNEUMONIA",
        "confidence": 0.91,
        "probabilities": {"NORMAL": 0.09, "PNEUMONIA": 0.91},
        "gradcam_score": 0.65,
    }

    metadata = select_demo_samples.build_metadata_entry(
        "high_confidence_pneumonia",
        candidate,
        "pneumonia-high-confidence.jpeg",
    )

    assert metadata["case_title"] == "High-confidence pneumonia"
    assert "confirmed" not in metadata["case_title"].lower()
    assert "Real model output with high pneumonia probability." == metadata["case_description"]
    assert metadata["gradcam_quality_score"] == 0.65


def test_build_metadata_keeps_actual_probability_values():
    candidate = {
        "source_path": "source.jpeg",
        "predicted_label": "PNEUMONIA",
        "confidence": 0.62,
        "probabilities": {"NORMAL": 0.38, "PNEUMONIA": 0.62},
    }

    metadata = select_demo_samples.build_metadata_entry(
        "borderline",
        candidate,
        "borderline-pneumonia.jpeg",
    )

    assert metadata["case_title"] == "Borderline case"
    assert metadata["filename"] == "borderline-pneumonia.jpeg"
    assert metadata["source_path"] == "source.jpeg"
    assert metadata["predicted_label"] == "PNEUMONIA"
    assert metadata["confidence"] == 0.62
    assert metadata["normal_probability"] == 0.38
    assert metadata["pneumonia_probability"] == 0.62
    assert "Near the model decision boundary" in metadata["case_description"]


def test_selector_uses_one_service_and_recovers_from_corrupt_image(tmp_path, monkeypatch):
    settings_seen, images_seen = [], []

    class Service:
        def __init__(self, settings):
            settings_seen.append(settings)

        def predict(self, image):
            images_seen.append((image.mode, image.size))
            return Prediction("NORMAL", 0.8, 0.8, 0.2)

    monkeypatch.setattr("src.inference.PredictionService", Service)
    valid, corrupt = tmp_path / "valid.png", tmp_path / "corrupt.png"
    Image.new("RGB", (40, 30)).save(valid)
    corrupt.write_bytes(b"corrupt")
    config = {"checkpoint_path": tmp_path / "custom.pth", "device": "cpu"}
    candidates = select_demo_samples.predict_candidates([valid, corrupt, valid], config)
    assert len(candidates) == 2 and len(settings_seen) == 1
    assert settings_seen[0].device == "cpu"
    assert settings_seen[0].checkpoint_path == config["checkpoint_path"]
    assert images_seen == [("L", (40, 30)), ("L", (40, 30))]


def test_sample_metadata_is_written_beside_custom_output_images(tmp_path):
    source = tmp_path / "source.png"
    Image.new("L", (10, 10)).save(source)
    candidate = {
        "source_path": str(source),
        "predicted_label": "NORMAL",
        "confidence": 0.8,
        "probabilities": {"NORMAL": 0.8, "PNEUMONIA": 0.2},
    }
    output = tmp_path / "samples"
    select_demo_samples.write_selected_samples({"normal": candidate}, output)
    assert (output / "demo_samples.json").is_file()
    assert (output / "normal-example.jpeg").read_bytes() == source.read_bytes()


def test_selector_entry_point_uses_configured_dataset_and_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "custom.pth"
    checkpoint.write_bytes(b"preflight checks existence only")
    data_dir = tmp_path / "custom-data"
    monkeypatch.setattr(
        select_demo_samples,
        "load_config",
        lambda: {
            "checkpoint_path": checkpoint,
            "data_dir": data_dir,
        },
    )
    searched = []
    monkeypatch.setattr(
        select_demo_samples, "discover_images", lambda paths: searched.extend(paths) or []
    )
    assert select_demo_samples.main() == 1
    assert searched == [
        data_dir / split / label
        for split in ("test", "val", "train")
        for label in ("NORMAL", "PNEUMONIA")
    ]
