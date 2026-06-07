import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "demo-assets" / "sample-xrays"
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
    samples = {}
    if not sample_dir.exists():
        return samples

    for path in sorted(sample_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_text = str(path.relative_to(sample_dir)).upper()
        if "NORMAL" in label_text and "NORMAL" not in samples:
            samples["NORMAL"] = path
        elif "PNEUMONIA" in label_text and "PNEUMONIA" not in samples:
            samples["PNEUMONIA"] = path
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
        return Image.open(uploaded_file).convert("L"), "Uploaded image"
    if selected_sample is not None:
        return Image.open(selected_sample).convert("L"), selected_sample.name
    return None, None


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
        st.caption("Use a local demo sample for the reel, or upload a chest X-ray image.")
        uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

        st.markdown("#### Built-in samples")
        if samples:
            sample_cols = st.columns(2)
            if "NORMAL" in samples and sample_cols[0].button("Normal example", use_container_width=True):
                st.session_state["selected_sample"] = str(samples["NORMAL"])
            if "PNEUMONIA" in samples and sample_cols[1].button("Pneumonia example", use_container_width=True):
                st.session_state["selected_sample"] = str(samples["PNEUMONIA"])
        else:
            st.info(
                "Add demo images under demo-assets/sample-xrays/. "
                "Use filenames or subfolders containing NORMAL and PNEUMONIA."
            )

        selected_sample = st.session_state.get("selected_sample")
        if selected_sample:
            st.caption(f"Selected sample: {Path(selected_sample).name}")

    image, image_caption = load_selected_image(uploaded_file, Path(selected_sample) if selected_sample else None)

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
                    from src.gradcam import generate_gradcam_overlay
                    from src.inference import predict_image

                    st.session_state["prediction"] = predict_image(image, config)
                    try:
                        st.session_state["gradcam_overlay"] = generate_gradcam_overlay(image, config)
                        st.session_state["gradcam_error"] = None
                    except Exception as exc:
                        st.session_state["gradcam_overlay"] = None
                        st.session_state["gradcam_error"] = str(exc)

                prediction = st.session_state.get("prediction")
                if prediction:
                    st.metric("Prediction", prediction["label"], f"{prediction['confidence']:.1%} confidence")
                    render_probability_bars(prediction["probabilities"])
                else:
                    st.caption("Run prediction to show confidence, probabilities, and explanation.")

    if image is not None:
        overlay = st.session_state.get("gradcam_overlay")
        gradcam_error = st.session_state.get("gradcam_error")
        st.markdown('<div id="model-explanation-anchor"></div>', unsafe_allow_html=True)
        st.markdown("### Model explanation")
        st.caption("Highlighted regions influenced the prediction.")
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
