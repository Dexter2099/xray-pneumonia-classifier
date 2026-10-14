# Working on the X-ray classifier

Read `README.md` and `docs/architecture.md` for the application boundaries. For code, tests or verification changes, use `.agents/skills/xray-maintenance/SKILL.md`.

## Scientific invariants

- Preserve the existing CNN, class order (`NORMAL=0`, `PNEUMONIA=1`), 224 x 224 grayscale input and `(value - 0.5) / 0.5` normalisation unless the task explicitly changes the scientific model.
- Keep `tests/fixtures/trained_baseline.json` and `legacy_inference.py.txt` as historical references. Never regenerate or loosen them just to make a refactor pass. Model/preprocessing changes require a separately explained validation decision.
- Reuse `PredictionService` for repeated calls; the compatibility helper is deliberately one-shot. Use `src/inputs.py` for the shared input contract.
- Prediction disables gradients; Grad-CAM enables gradients in its own pass, removes hooks and clears intermediates on success and failure. Serialize access to the shared model.
- The desktop worker owns expensive model/image operations. Widgets remain on the GUI thread. New inputs clear stale results; explanation failures retain successful predictions.

## Verification and delivery

- Use Python 3.11 and the isolated `.venv`; tested dependencies are pinned in `requirements-viewer.txt` and `requirements-dev.txt`.
- Run `python -m scripts.check` for the portable checks. Use `--trained` for inference, preprocessing, Grad-CAM or desktop changes when the trusted local checkpoint is available. Use `--release` before an authorized merge to main in this equipped workspace.
- Ordinary CI must work without the dataset/checkpoint. Explicitly requested trained checks fail when their inputs are missing or changed; do not turn that into a silent skip.
- Review the diff and the test intent. Keep commits focused on a behavior or design change and explain the reason and evidence in the PR. Use component milestones, not calendar-day labels.
- Push or merge main only within the user's authorization. Existing authorization persists; do not introduce a second approval step. Do not force-push, bypass protection, or claim pending/unrun remote checks passed.
- Keep datasets, checkpoints, generated recordings, private images, credentials and logs out of Git. The existing public demo fixtures are deliberate; don't add new images as part of routine implementation.
- Distinguish AI drafts, automated review, executed verification and human acceptance. Never invent manual testing, production experience, scientific validity or AI contribution.

## Workspace memory

Follow attached PMB instructions when those tools are available, including preparation, surfaced-lesson follow-through and one triggered log batch per turn. If the tools are unavailable, state that limitation and continue from verified repository evidence; do not invent memory calls or install/configure a server without a relevant request.
