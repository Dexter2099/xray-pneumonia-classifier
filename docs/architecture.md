# Design decisions

The existing CNN and its checkpoint remain the scientific baseline. Engineering changes make its inputs, lifetime and failure behaviour explicit. A future model or preprocessing change needs a new scientific validation exercise rather than updating golden expectations to make tests pass.

## Responsibilities and dependencies

| Component | Responsibility |
|---|---|
| `src/inputs.py` | Decode bounded PNG/JPEG inputs and produce the model's canonical tensor. |
| `src/settings.py` | Validate checkpoint/device settings and anchor paths to the repository. |
| `PredictionService` | Load a trusted state dictionary once, own its device and return typed predictions. |
| `GradCamExplainer` | Generate an explanation with the existing model and release hooks/graphs. |
| `AnalysisWorker` | Own a backend in one Qt worker thread and serialize expensive operations. |
| `ViewerController` | Manage busy/ready/error state and associate results with requests. |
| `ViewerWindow` | Render pixels and text and emit user actions; it imports no model code. |
| `InputInspection` / `TimelineEvent` | Small detached snapshots and actual operation events; no retained tensors or graphs. |
| `UnderTheHood` | Optional dock rendering input statistics, worker events and saved evidence. |
| `read_evidence` | Bound JSON input, compare saved expectations/tolerances with the frozen baseline, recompute differences and label provenance. |

## SOLID applied to this application

- **Single responsibility:** image validation, scientific computation, background execution and presentation are separate.
- **Open/closed:** a backend factory can supply another predictor or explainer without changing the controller's workflow. This is an extension point, not a claim that arbitrary checkpoints are supported.
- **Liskov substitution:** the real predictor and deterministic test predictor satisfy the same `predict(source) -> Prediction` contract. The GUI workflow tests run through that substituted backend.
- **Interface segregation:** `Predictor` and `Explainer` are separate small protocols. Prediction does not require an explanation method or GUI dependency.
- **Dependency inversion:** the worker accepts a backend factory. The controller does not construct a CNN, deserialize weights or implement preprocessing. No dependency-injection framework is needed.

## Why model lifetime belongs to a service

The legacy helper constructed and loaded a model for every image. `PredictionService` has an explicit lifetime and can be reused across predictions and Grad-CAM. The desktop worker creates one service; Streamlit uses a resource cache keyed by checkpoint path/size/modification time. A reentrant lock serializes access when a service is shared. Desktop requests are processed sequentially.

`predict_image` remains a one-shot compatibility adapter. Existing scripts and notebooks keep working; repeated consumers should inject a service or use `service.predict` directly. There is no hidden process-global model cache.

## Why prediction and explanation use different gradient contexts

Prediction uses evaluation mode plus `torch.inference_mode()`. Grad-CAM also uses evaluation mode, but explicitly enables gradients for a separate forward/backward pass. Its hooks exist only inside `generate` and are removed in `finally`; stored tensors and parameter gradients are cleared on success and failure. No graph is retained between explanations.

## Failure and state behaviour

Expected failures use application exceptions. The GUI displays actionable messages. Unexpected failures are caught at the thread boundary so they do not strand a disabled window; diagnostics go to standard logging. Images are never written to logs.

A new image clears the previous prediction and overlay before decoding. Corrupt files leave no stale input/results. Request identifiers reject stale callbacks. Conflicting actions are disabled and checked again by the controller. If Grad-CAM fails, the completed prediction remains visible. Closing during work waits for that operation before the worker thread exits; it never forcibly terminates a PyTorch operation.

## Optional diagnostics preserve the inference boundary

The input inspector calls the existing `preprocess_image` function on the validated grayscale image to build bounded previews and scalar statistics. It does not intercept, modify or substitute the tensor used by `PredictionService`. Reverse normalisation is used only to render the resized preview. The scientific model, input contract and prediction/explanation paths are unchanged.

The worker measures real operations and sends immutable event metadata through queued Qt signals. Prediction remains one combined preprocessing/forward/softmax stage; Grad-CAM remains a separate gradient-enabled pass. The timeline does not fabricate operator-level measurements. It retains at most 100 stages and never contains image pixels. Diagnostics are collected while the panel is hidden, with their overhead shown separately from the inference benchmark. Inspector/report failures are isolated so they cannot block an otherwise valid prediction or strand a busy window.

Verification refresh performs bounded file reads and Git identity checks in the worker, without loading another model or executing tests. Local `outputs/reports/regression.json` takes precedence over the committed historical snapshot. An invalid local report is shown as unavailable rather than silently replaced by an old PASS. The reader checks references and tolerances against the frozen baseline and recomputes per-case comparisons. A recorded PASS is evidence about a saved run; it is not a current prediction or cryptographic attestation of execution. Source/checkpoint matching is checked at refresh time, and a clean source revision is required before calling evidence current. Unidentified, dirty or different revisions remain historical.

## Scope and trade-offs

This is a source-launched desktop application, not a packaged installer. PNG/JPEG and the existing 224 × 224 two-class CNN are supported; DICOM, 16-bit image interpretation, retraining, calibration and dataset-leakage investigation are separate work. Input resizing and normalisation intentionally preserve the existing model's assumptions.
