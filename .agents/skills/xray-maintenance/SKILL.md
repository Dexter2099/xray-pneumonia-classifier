---
name: xray-maintenance
description: Implement or review code and verification changes in this X-ray classifier while preserving its scientific baseline and checking desktop behavior.
---

# Maintain the X-ray application

Read the repository's `AGENTS.md`, then inspect the affected component and its callers. Use `docs/architecture.md` for responsibilities and `docs/verification.md` for the meaning and limits of the evidence. These paths are relative to the repository root.

Define the concrete change and its compatibility boundary before editing. The current model expects one 224 x 224 grayscale channel, raw image/tensor values in [0,1], one normalisation to [-1,1], and fixed NORMAL/PNEUMONIA class order. A code cleanup must preserve predictions; new input rejection should be explicit.

Keep generated work bounded to the affected component. AI may draft implementation and tests, but inspect the resulting source, failure paths and assertions rather than accepting plausible code or a passing smoke test alone. If local Qwen is available, use it for suitable bounded drafts; this skill does not require downloading models or a paid API.

## Choose verification by the change

- Documentation only: check links, commands, factual claims and `git diff --check`. Avoid rerunning expensive model checks when no behavior changed.
- Application/test/tooling changes: run `python -m scripts.check` with the project environment.
- Inference, preprocessing, Grad-CAM or desktop behavior: also run `python -m scripts.check --trained` when the trusted checkpoint is available. Inspect the regression report and, for UI work, the generated desktop screenshots.
- Performance changes: add `--benchmark` to the trained run. Compare the frozen legacy path and current implementation under the same runtime/device, distinguish startup from warm prediction, and reject timing results if predictions disagree.
- Authorized merge/release in the equipped local workspace: run `python -m scripts.check --release`, inspect the final diff, and use the repository's ordinary PR workflow. Stop on failed gates and fix the cause; never refresh the golden baseline as a repair.

When local artifacts are unavailable, run portable CI checks and report the missing trained verification. Do not describe ordinary CI as a trained-model regression or an automated Qt run as human acceptance of the native interface.

Keep the existing model and baseline fixtures unless the user explicitly requests a scientific change. Put generated reports under ignored `outputs/` or `demo-output/`; commit selected evidence only after checking it contains no private inputs, credentials or machine-specific paths.

## Review and handoff

Explain the resulting behavior, one relevant design decision, checks actually run and material limitations. Record the work in focused commits and a reviewable PR. Do not imply the human wrote or personally reviewed every line merely because they directed AI-assisted work. See `docs/ai-assisted-development.md` for the project's actual AI contribution and harness history.
