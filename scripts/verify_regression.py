"""Verify independently captured trained-model outputs; never regenerate the baseline."""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from src.errors import XrayError
from src.inference import PredictionService
from src.provenance import source_state
from src.settings import ROOT, InferenceSettings


def verify(checkpoint: Path, threads: int = 4) -> dict:
    revision, dirty = source_state(ROOT)
    torch.set_num_threads(threads)
    baseline = json.loads((ROOT / "tests/fixtures/trained_baseline.json").read_text())
    service = PredictionService(InferenceSettings(checkpoint, "cpu"))
    if service.checkpoint_sha256 != baseline["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the independently captured baseline.")
    sources, cases = [], []
    for case in baseline["cases"]:
        path = ROOT / case["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("A baseline input image has changed.")
        sources.append(path)
        actual = service.predict(path).to_dict()
        expected = case["prediction"]
        values = list(actual["probabilities"].values())
        reference = list(expected["probabilities"].values())
        agrees = actual["label"] == expected["label"] and np.allclose(
            values,
            reference,
            atol=baseline["absolute_tolerance"],
            rtol=baseline["relative_tolerance"],
        )
        cases.append(
            {
                "image": case["image"],
                "passed": bool(agrees),
                "max_absolute_error": float(np.max(np.abs(np.array(values) - reference))),
                "expected": expected,
                "actual": actual,
            }
        )
    singles = [service.predict(path) for path in sources]
    batches = service.predict_batch(sources)
    batch_agrees = all(
        single.label == batch.label
        and np.allclose(
            [single.normal_probability, single.pneumonia_probability],
            [batch.normal_probability, batch.pneumonia_probability],
            atol=baseline["absolute_tolerance"],
            rtol=baseline["relative_tolerance"],
        )
        for single, batch in zip(singles, batches, strict=True)
    )
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "working_tree_dirty": dirty,
        "passed": all(case["passed"] for case in cases) and batch_agrees,
        "baseline_commit": baseline["baseline_commit"],
        "baseline_torch_version": baseline["torch_version"],
        "checkpoint_sha256": service.checkpoint_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": threads,
        },
        "absolute_tolerance": baseline["absolute_tolerance"],
        "relative_tolerance": baseline["relative_tolerance"],
        "single_batch_agreement": bool(batch_agrees),
        "model_loads": service.model_load_count,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/best_model.pth"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/regression.json")
    args = parser.parse_args()
    try:
        report = verify(args.checkpoint)
    except (XrayError, OSError, ValueError) as exc:
        parser.exit(1, f"Regression verification failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Regression {'PASS' if report['passed'] else 'FAIL'}: {len(report['cases'])} images; report: {args.output}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
