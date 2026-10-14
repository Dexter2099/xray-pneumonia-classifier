# X-ray Pneumonia Classifier

## Project Overview
This repository demonstrates maintaining an existing scientific Python application: a PyTorch CNN, validated image inputs, a reusable inference service, a responsive PySide6 desktop viewer and Grad-CAM visual explanations. The original model architecture and trained weights are preserved, with independently captured regression results and measured performance improvements.

This is educational portfolio software, not a diagnostic system. Model probabilities are not calibrated clinical confidence, and Grad-CAM highlights model influence rather than disease evidence.

## Desktop viewer

Use Python 3.11. From the repository root, create an isolated environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m src.viewer
```

The trusted local state dictionary must exist at `outputs/checkpoints/best_model.pth`. Missing or incompatible checkpoints produce a recoverable message in the viewer. Place the checkpoint at the configured location, then use **Reload model**. To change the configuration path or device, edit `src/config.yaml` and restart the viewer. Checkpoints and datasets are not committed or downloaded automatically.

The viewer validates PNG/JPEG files before prediction, displays the original image and explanation, keeps long operations off the GUI thread, and shows prediction timing and model-load counters. Model loading occurs once per session unless explicitly reloaded. A corrupt image clears stale results and leaves the application ready to open another image.

Select **Under the hood** to open an optional dock with three tabs:

- **Input inspector:** decoded grayscale and resized previews, tensor shape/dtype, numerical statistics and a histogram. It previews the shared preprocessing function separately; it does not replace the inference tensor or retain a computation graph.
- **Execution timeline:** actual worker stages, request IDs, outcomes, durations and thread IDs. Corrupt input stops at decoding; failed explanations retain successful predictions. The last 100 stages are kept in memory.
- **Verification:** saved original-versus-refactored probabilities, numerical differences, frozen tolerances, checkpoint identity and source revision. Generate fresh evidence with `python -m scripts.verify_regression`, then select **Refresh saved report**. Refresh reads evidence without running tests. Historical, dirty or unidentified source states and different active checkpoints are explicitly labelled.

The panel starts hidden, can be closed or detached, and does not trigger model loading or prediction when toggled. Small bounded diagnostics are captured in the worker even while hidden so opening it shows the current input and recent operations. The inspector starts after grayscale decoding; it does not claim to preserve original colour. Stage timings include diagnostics overhead and are separate from the controlled benchmark.

Use **Text size → Normal / Large** for 12-point or 15-point body text throughout the viewer and diagnostic panels. Prediction headings are larger, status/error text has stronger contrast, and tables have adjustable columns and readable row heights. Resizing gives additional space to the images; compact windows and long input-inspector content can scroll. Text-size changes are session-only and do not reload the model or change predictions.

Files are limited to 20 MiB and 20 million pixels. Images become 8-bit grayscale and resize to 224 × 224. Raw tensor inputs must be dense float32, shaped `(1, 224, 224)`, finite and in `[0,1]`; normalisation then maps them to `[-1,1]` once. Configuration paths resolve relative to the repository, independently of the launch directory. The model's two-logit argmax rule is preserved; the old YAML `threshold` value is not used.

## Verification and interview walkthrough

The repository's [AI-assisted engineering harness](docs/ai-assisted-development.md) combines `AGENTS.md`, a discoverable maintenance skill, reference fixtures, focused tests and an executable verification command:

```powershell
.venv\Scripts\python.exe -m scripts.check
.venv\Scripts\python.exe -m scripts.check --release
```

The release profile adds trained-model regression, the real desktop workflow, benchmark and dependency audit. It fails on an unsuccessful gate and records what ran in `outputs/verification/checks.json`. See [the maintenance skill](.agents/skills/xray-maintenance/SKILL.md) for selecting checks by the change.

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -q --run-trained
.venv\Scripts\python.exe -m scripts.verify_regression
.venv\Scripts\python.exe -m scripts.benchmark_inference
.venv\Scripts\python.exe -m scripts.viewer_smoke
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy
.venv\Scripts\python.exe -m pip_audit -r requirements-viewer.txt
```

Ordinary tests use synthetic images/checkpoints. `--run-trained` explicitly requires the original local checkpoint and three fingerprinted demo images; a missing or changed checkpoint fails the requested verification. The baseline was captured before refactoring, rather than generated from the new implementation.

The recorded CPU benchmark measured **84.8 ms → 7.9 ms** median prediction time across seven repeats, with identical output probabilities. Both implementations used the current runtime and warm filesystem cache. Model startup was measured separately; timings exclude Python imports and file decoding and will vary by machine.

- [Five-section interview demonstration](docs/interview-demo.md)
- [Architecture, SOLID boundaries and design decisions](docs/architecture.md)
- [Verification evidence and limitations](docs/verification.md)
- [Original-to-refactored prediction regression](docs/evidence/regression.json)
- [Benchmark measurements](docs/evidence/benchmark.json)
- [624-image evaluation report](docs/evidence/evaluation.json)

The component history is preserved in focused PRs: [existing demo baseline #4](https://github.com/Dexter2099/xray-pneumonia-classifier/pull/4), [inference core #5](https://github.com/Dexter2099/xray-pneumonia-classifier/pull/5), then [desktop viewer #6](https://github.com/Dexter2099/xray-pneumonia-classifier/pull/6). They began as a stack and were retargeted to main as their dependencies merged; the merge commits retain the focused implementation commits.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Dexter2099/xray-pneumonia-classifier.git
   cd xray-pneumonia-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Chest X-ray dataset from Kaggle using the [Kaggle API](https://github.com/Kaggle/kaggle-api) or from [Kaggle - Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Extract the `chest_xray` folder into `data/raw/` so that the path looks like `data/raw/chest_xray/train/...`. The dataset is large and is **not included** in this repository.

## Training
Train the model from the command line:
```bash
python main.py --mode train
```
Model checkpoints are saved to `outputs/checkpoints/`.

## Evaluation
After training, evaluate the model:
```bash
python main.py --mode eval
```
Evaluation metrics and confusion matrix images will be written to `outputs/visualizations/`.

## Grad-CAM demo
Run Grad-CAM on a specific image to visualize what the model focuses on:
```bash
python main.py --mode gradcam --image path/to/image.jpeg
```
The resulting heatmap is stored in `outputs/visualizations/`.

## Notebook walkthrough
- `notebooks/01_EDA.ipynb` contains a short exploratory data analysis.
- `notebooks/02_Model_Demo.ipynb` offers a guided demo showing how to load the trained model and run predictions on sample images.

---
Created as a beginner-friendly yet medically relevant deep learning project.

## Web App
You can launch a simple Streamlit interface to test the model on your own images:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Portfolio demo reel
The Streamlit app includes a portfolio demo mode for recording a short product-style reel. It keeps the project in Streamlit and uses the local trained checkpoint, local demo X-ray samples, and existing evaluation artifacts.

This project is educational portfolio software only. It is not clinical software and must not be used for medical diagnosis.

### Expected local files
The selector and recorder expect these files to exist locally:

```text
outputs/checkpoints/best_model.pth
data/raw/chest_xray/
```

Generate the three demo cases from real model outputs:

```bash
python scripts/select_demo_samples.py
```

The selector scans local X-ray images, runs the trained checkpoint, and writes:

```text
demo-assets/sample-xrays/demo_samples.json
demo-assets/sample-xrays/pneumonia-high-confidence.jpeg
demo-assets/sample-xrays/normal-example.jpeg
demo-assets/sample-xrays/borderline-pneumonia.jpeg
```

The high-confidence pneumonia case is selected from real model outputs using both pneumonia probability and Grad-CAM visual quality, so the reel shows a prediction that is also explainable on screen. The borderline case is selected from real model probabilities. It may not be exactly 50%; the metadata records the actual NORMAL and PNEUMONIA probabilities. Grad-CAM highlights model-influential regions, not a clinical diagnosis.

Large datasets and model checkpoints should stay local. Do not commit `data/`, `outputs/`, or large model files.

### Run the app
```bash
streamlit run app.py
```

The demo page supports uploaded images and three built-in real-model cases: high-confidence pneumonia, normal X-ray, and borderline case. It shows the original X-ray, prediction label, confidence score, class probability bars, Grad-CAM overlay, and the confusion matrix when `outputs/visualizations/confusion_matrix.png` exists. Metrics are only shown when a real metrics JSON file exists.

### Record the reel
Install Playwright's Chromium browser once:

```bash
python -m playwright install chromium
```

Then make sure samples are selected and record the demo:

```bash
python scripts/select_demo_samples.py
python scripts/record_demo.py
```

The recorder launches Streamlit on a free local port, walks through all three demo cases in a 1440x900 browser viewport, records with Playwright page-video capture, and writes:

```text
demo-output/xray-classifier-demo.mp4
```

If the checkpoint, `demo_samples.json`, or one of the three required sample images is missing, the recorder exits with a clear message telling you to run `python scripts/select_demo_samples.py`.
