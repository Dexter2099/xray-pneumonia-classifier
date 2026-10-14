import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image
from src.errors import XrayError
from src.inputs import load_image
from src.settings import InferenceSettings


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "demo-assets" / "sample-xrays"
DEMO_METADATA_PATH = SAMPLE_DIR / "demo_samples.json"
GRADCAM_OUTPUT_DIR = ROOT / "demo-output" / "gradcam"
CONFUSION_MATRIX_PATH = ROOT / "outputs" / "visualizations" / "confusion_matrix.png"
METRICS_CANDIDATES = [
    ROOT / "outputs" / "visualizations" / "metrics.json",
    ROOT / "outputs" / "metrics.json",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_config(config_path=ROOT / "src" / "config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def find_demo_samples(sample_dir=SAMPLE_DIR):
    metadata_path = sample_dir / "demo_samples.json"
    if not metadata_path.exists():
        return []

    samples = json.loads(metadata_path.read_text(encoding="utf-8"))
    for sample in samples:
        sample["path"] = sample_dir / sample["filename"]
    return samples


def load_metrics():
    for path in METRICS_CANDIDATES:
        if path.exists():
            return pd.read_json(path, typ="series").to_dict()
    return None


def render_probability_bars(probabilities):
    for label, probability in probabilities.items():
        st.write(f"{label}: {probability:.1%}")
        st.progress(probability)


def load_selected_image(uploaded_file, selected_sample):
    if uploaded_file is not None:
        return load_image(uploaded_file.getvalue()), "Uploaded image", None
    if selected_sample is not None:
        path = Path(selected_sample["path"])
        return load_image(path), selected_sample["case_title"], selected_sample
    return None, None, None


def selected_sample_from_state(samples):
    selected_case_id = st.session_state.get("selected_case_id")
    if selected_case_id is None and samples:
        selected_case_id = samples[0]["case_id"]
        st.session_state["selected_case_id"] = selected_case_id
    for sample in samples:
        if sample["case_id"] == selected_case_id:
            return sample
    return None


def gradcam_cache_path(sample):
    if sample is None:
        return None
    return GRADCAM_OUTPUT_DIR / f"{sample['case_id']}.png"


@st.cache_resource
def cached_prediction_service(checkpoint_path, checkpoint_size, checkpoint_mtime):
    from src.inference import PredictionService

    return PredictionService(InferenceSettings(Path(checkpoint_path)))


def load_or_generate_gradcam(image, config, sample, service=None):
    from src.gradcam import generate_gradcam_overlay

    # Recompute with the shared model; case-only disk caches can become stale.
    return generate_gradcam_overlay(image, config, service=service)


def main():
    st.set_page_config(
        page_title="Chest X-Ray Pneumonia Classifier",
        page_icon="X",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.35rem; max-width: 1160px; }
        .demo-disclaimer {
            border-left: 4px solid #d97706;
            background: #fff7ed;
            padding: 0.55rem 0.8rem;
            margin: 0.75rem 0 0.7rem;
        }
        h1 { margin-bottom: 0.2rem; }
        h2, h3 { margin-top: 0.45rem; }
        div[data-testid="stVerticalBlock"] { gap: 0.45rem; }
        div[data-testid="stImage"] img { max-height: 315px; object-fit: contain; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Chest X-Ray Pneumonia Classifier")
    st.subheader("PyTorch CNN with Grad-CAM visual explanation")
    st.markdown(
        '<div class="demo-disclaimer">Portfolio demo only. Not for clinical use.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.36, 0.64], gap="large")
    samples = find_demo_samples()
    config = load_config()

    with left:
        st.markdown("### Demo input")
        st.caption("Use a selected real-model demo case, or upload a chest X-ray image.")
        uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

        st.markdown("#### Demo cases")
        if samples:
            for sample in samples:
                button_label = sample["case_title"]
                if st.button(button_label, use_container_width=True):
                    st.session_state["selected_case_id"] = sample["case_id"]
                    st.session_state.pop("prediction", None)
                    st.session_state.pop("gradcam_overlay", None)
                    st.session_state.pop("gradcam_error", None)
        else:
            st.info(
                "Run `python scripts/select_demo_samples.py` to create the three real-model demo cases."
            )

        selected_sample = selected_sample_from_state(samples)
        if selected_sample:
            st.caption(f"Selected case: {selected_sample['case_title']}")
            st.caption(selected_sample["case_description"])

    try:
        image, image_caption, selected_sample = load_selected_image(uploaded_file, selected_sample)
    except XrayError as exc:
        st.error(str(exc))
        image, image_caption = None, None
    input_key = None if image is None else (image.size, image.tobytes())
    if input_key != st.session_state.get("input_key"):
        for key in ("prediction", "gradcam_overlay", "gradcam_error"):
            st.session_state.pop(key, None)
        st.session_state["input_key"] = input_key

    with right:
        st.markdown("### Prediction workflow")
        if image is None:
            st.info("Choose a built-in sample or upload an image to start the demo.")
        else:
            image_col, result_col = st.columns([0.48, 0.52], gap="large")
            with image_col:
                st.image(image, caption=f"Original X-ray: {image_caption}", use_container_width=True)

            with result_col:
                if st.button("Run prediction", type="primary", use_container_width=True):
                    from src.inference import predict_image

                    try:
                        settings = InferenceSettings.from_mapping(config)
                        stat = settings.checkpoint_path.stat()
                        service = cached_prediction_service(str(settings.checkpoint_path), stat.st_size, stat.st_mtime_ns)
                        st.session_state["prediction"] = predict_image(image, config, service=service)
                    except (XrayError, OSError) as exc:
                        st.error(f"Prediction unavailable: {exc}")
                        st.session_state.pop("prediction", None)
                        st.session_state.pop("gradcam_overlay", None)
                        st.session_state.pop("gradcam_error", None)
                        return
                    try:
                        st.session_state["gradcam_overlay"] = load_or_generate_gradcam(image, config, selected_sample, service)
                        st.session_state["gradcam_error"] = None
                    except Exception as exc:
                        st.session_state["gradcam_overlay"] = None
                        st.session_state["gradcam_error"] = str(exc)

                prediction = st.session_state.get("prediction")
                if prediction:
                    st.metric("Prediction", prediction["label"], f"{prediction['confidence']:.1%} confidence")
                    if selected_sample and selected_sample["case_id"] == "borderline":
                        st.caption("Borderline model output. This case sits near the model decision boundary.")
                    render_probability_bars(prediction["probabilities"])
                else:
                    st.caption("Run prediction to show confidence, probabilities, and explanation.")

    if image is not None:
        overlay = st.session_state.get("gradcam_overlay")
        gradcam_error = st.session_state.get("gradcam_error")
        st.markdown('<div id="model-explanation-anchor"></div>', unsafe_allow_html=True)
        st.markdown("### Model explanation")
        st.caption("Highlighted regions influenced the model prediction. For normal cases, this is not disease evidence.")
        gradcam_cols = st.columns(2, gap="medium")
        gradcam_cols[0].image(image.resize((260, 260)), caption="Original X-ray", width=260)
        if overlay is not None:
            gradcam_cols[1].image(
                overlay.resize((260, 260)),
                caption="Grad-CAM overlay",
                width=260,
            )
        elif gradcam_error:
            gradcam_cols[1].warning(f"Grad-CAM unavailable: {gradcam_error}")
        else:
            gradcam_cols[1].info("Grad-CAM overlay appears after prediction.")

    show_evaluation = os.environ.get("XRAY_DEMO_RECORDING") != "1"
    if show_evaluation and (CONFUSION_MATRIX_PATH.exists() or load_metrics()):
        st.divider()
        st.markdown("### Evaluation artifacts")
        eval_cols = st.columns(2, gap="large")
        if CONFUSION_MATRIX_PATH.exists():
            eval_cols[0].image(str(CONFUSION_MATRIX_PATH), caption="Confusion matrix", use_container_width=True)
        metrics = load_metrics()
        if metrics:
            with eval_cols[1]:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(str(key).replace("_", " ").title(), f"{value:.3f}")
                    else:
                        st.write(f"**{key}:** {value}")


if __name__ == "__main__":
    main()
