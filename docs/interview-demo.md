# Interview demonstration

Prepare the trusted checkpoint at `outputs/checkpoints/best_model.pth`. From the repository root:

```powershell
.venv\Scripts\python.exe -m scripts.viewer_smoke
.venv\Scripts\python.exe -m src.viewer
```

The smoke run creates a deliberately corrupt PNG at `demo-output/viewer/corrupt-example.png`, screenshots and a JSON result. It uses the actual checkpoint and desktop worker; it is an automated workflow check rather than human acceptance of the native interface.

Before presenting, maximize the viewer and choose **Text size → Large** if needed. The control applies to the diagnostic panels as well as the main window. Switching sizes preserves the current image, result and model-load count. Diagnostic tables allow column resizing; long details and compact windows scroll rather than reducing the font.

## 1. Open an image and show prediction and explanation

Open `demo-assets/sample-xrays/pneumonia-high-confidence.jpeg`, then select **Predict and explain**. Show the prediction, class probabilities and Grad-CAM overlay.

Open **Under the hood → Input inspector**. Show the original grayscale dimensions, the 224 × 224 resized preview, float32 shape, normalised range and histogram. Explain that the inspector uses the shared preprocessing function in a separate diagnostic call; the inference service still prepares its own input.

Say: “I kept the existing trained model. The improvement is how the application validates inputs, runs it and handles failure. The heatmap shows regions influencing the output; it does not establish a diagnosis.”

## 2. Open a corrupt image and demonstrate recovery

Open `demo-output/viewer/corrupt-example.png`. Show the message, cleared previous result and disabled prediction action. Reopen the valid image and run a successful prediction.

In **Execution timeline**, show the failed **Decode / validate** stage and the later successful request. The input inspector is cleared when decoding fails. Click a stage to read its responsibility.

Say: “Image decoding happens at an explicit boundary. A corrupt file produces a useful error, and the application remains usable without restarting.”

## 3. Run repeated predictions and show model reuse

Select **Predict and explain** twice. Point to **Model loads: 1** and the increasing prediction counter. Move or resize the window while processing.

Show the request IDs, prediction and Grad-CAM stages and common worker-thread ID in the timeline. Prediction uses evaluation mode with gradients disabled; Grad-CAM requires a separate gradient-enabled pass. These are operation timings, not operator profiling or the controlled benchmark.

Say: “The worker owns one loaded model. Prediction and explanation reuse it sequentially, so the GUI stays responsive and Grad-CAM hooks cannot race with another request.”

## 4. Show regression results and measured performance

```powershell
.venv\Scripts\python.exe -m pytest -q --run-trained
.venv\Scripts\python.exe -m scripts.verify_regression
.venv\Scripts\python.exe -m scripts.benchmark_inference
```

Open `docs/evidence/regression.json` and `docs/evidence/benchmark.json`, or the fresh results in `outputs/reports/`. Explain that checkpoint/input hashes protect the baseline. The borderline case is included because tiny unintended changes could affect its decision.

Alternatively, select **Under the hood → Verification → Refresh saved report** after generating the regression report. Show expected versus recorded actual probabilities and maximum differences. The panel identifies the report's checkpoint, source revision, clean/dirty state and tolerances; old evidence stays labelled historical. Refreshing reads saved JSON and does not rerun tests or test the image currently open.

The recorded benchmark was 84.8 ms versus 7.9 ms median latency (10.72×), with identical probabilities. Both algorithms used the same current runtime, decoded image and CPU thread count. Startup is separate and the filesystem cache is warm; the benchmark is not a claim about every machine.

Mention the evaluation limitations if asked: the existing model achieved 78.53% accuracy on 624 test images, but NORMAL recall was only 44.44%. Preserving its behaviour does not make it clinically reliable.

## 5. Open a focused PR and explain a design decision

Open [inference core PR #5](https://github.com/Dexter2099/xray-pneumonia-classifier/pull/5). Show the ordered commits and the separation between `inputs.py`, `PredictionService` and `GradCamExplainer`. Then show [desktop PR #6](https://github.com/Dexter2099/xray-pneumonia-classifier/pull/6) and its worker/controller boundary.

Say: “I separated model lifetime from individual predictions. The controller receives a small backend contract, which lets me test corrupt-input recovery and UI state without loading the real model in every GUI test. I still run a separate regression against the actual trained checkpoint.”

If asked about AI, use the evidence and wording in [AI-assisted development and the harness](ai-assisted-development.md). Show `AGENTS.md`, `.agents/skills/xray-maintenance/SKILL.md` and `python -m scripts.check --release`. Explain that Codex and local Qwen contributed code, while the verification uses explicit contracts, reference results, tests and reviewable diffs. Distinguish automated source review from what you have personally inspected and rehearsed.
