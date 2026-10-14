# Verification evidence

Evidence captured on October 15, 2026, on this Windows machine with Python 3.11.9 and CPU PyTorch 2.14.1. Local measurements are snapshots, not performance guarantees.

| Check | Evidence |
|---|---|
| Original behaviour | `tests/fixtures/trained_baseline.json` captured from commit `fe9c3d2` before inference changes, using PyTorch 2.7.0. |
| Trained regression | Three fingerprinted images including the borderline case; unchanged labels/probabilities within absolute `1e-6`, relative `1e-5` tolerance. Single/batch agreement also verified. |
| Model reuse | Loader-spy test plus actual desktop run: three successful predictions and one model load. |
| Error recovery | Invalid tensor/configuration/image/checkpoint tests; corrupt-image GUI recovery; prediction retained when explanation fails. |
| Resource lifecycle | Hook/activation/gradient cleanup on repeated Grad-CAM calls and failures; orderly worker shutdown during processing. |
| Responsiveness | Qt timer continues firing during background prediction; model work verified to execute outside the GUI thread. |
| Optional diagnostics | Known pixel values/statistics, bounded previews/history, actual ordered worker events, stale-input clearing and diagnostic-failure isolation. Real desktop run compares predictions with the dock hidden and visible. |
| Evidence presentation | Malformed/oversized JSON, changed expectations/tolerances, inconsistent PASS flags, duplicate cases, nonfinite probabilities and stale/dirty/unknown source states are checked explicitly. Refresh reads evidence without running inference. |
| Performance | Seven alternating-order repeats, median 84.8 ms legacy vs 7.9 ms reused; maximum probability difference zero. Startup and Grad-CAM measured separately. |
| Evaluation | 624 test images, counts NORMAL=234/PNEUMONIA=390; confusion matrix `[[104,130],[4,386]]`, matching the pre-existing saved matrix. |

The ordinary CI suite uses synthetic inputs and checkpoints. The trained checkpoint and complete dataset remain local; requesting `--run-trained` requires the exact originals and fails if they are missing or changed. Committed evidence contains metrics, hashes and known demo-image identifiers, not model weights or new patient images.

Runtime dependencies are pinned in `requirements-viewer.txt`; developer tools in `requirements-dev.txt`. The original PyTorch/Pillow versions had reported advisories, so the project environment was upgraded and the original model regression rerun. The resolved runtime dependency audit reported no known vulnerabilities at verification time. This is an advisory-database result, not a guarantee that software has no vulnerabilities.

Ruff and focused mypy checks cover the modernized modules; the repository's older notebooks/training code are not presented as fully typed. The development virtual environment is isolated from global Python packages. The core Actions workflow covers Windows and Linux; actual remote results should be checked in the PR before merging.

The model's measured test performance is limited: 78.53% accuracy, NORMAL recall 44.44%, PNEUMONIA recall 98.97%. These metrics explain the existing model, not clinical utility. This work does not assess patient overlap, remove dataset leakage, calibrate probabilities or establish medical validity.

The automated desktop run and screenshots verify a real Qt workflow with the trained checkpoint. The human-operated native file dialog, display scaling and interview presentation still deserve a rehearsal on the actual screen.

Fresh regression reports now include a UTC generation time, source revision and working-tree state. The verification panel shows saved comparisons against the frozen baseline and active checkpoint, with source state sampled at refresh. This metadata identifies a recorded run; it does not authenticate the JSON or establish that tests are still passing after later edits. The committed evidence predates those provenance fields and therefore appears historical. `scripts.viewer_smoke` also renders all three optional tabs and checks unchanged predictions before/after opening the dock and recovering from corrupt input.

GitHub Actions now runs on the merged core and the viewer PR. The first Linux run identified a missing `libEGL.so.1` system dependency; the workflow now installs the Qt runtime libraries explicitly. Check the latest PR/commit status for the completed Windows/Linux result. Ordinary CI uses synthetic checkpoints; trained-model verification remains a separate local release check.
