"""Compare the frozen legacy algorithm and model reuse under identical current runtime settings."""

import argparse
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCamExplainer
from src.inference import PredictionService
from src.settings import ROOT, InferenceSettings


def benchmark(checkpoint: Path, repeats: int = 7, threads: int = 4) -> dict:
    if repeats < 3 or threads < 1:
        raise ValueError("Use at least three repeats and a positive thread count.")
    torch.set_num_threads(threads)
    namespace: dict = {}
    legacy_path = ROOT / "tests/fixtures/legacy_inference.py.txt"
    exec(compile(legacy_path.read_text(encoding="utf-8-sig"), str(legacy_path), "exec"), namespace)
    legacy_predict = namespace["predict_image"]
    config = {"checkpoint_dir": str(checkpoint.parent)}
    settings = InferenceSettings(checkpoint, "cpu")
    image_path = ROOT / "demo-assets/sample-xrays/pneumonia-high-confidence.jpeg"
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    # Excludes imports and image decoding from both paths; startup includes model construction/loading/hash.
    started = perf_counter()
    service = PredictionService(settings)
    startup = perf_counter() - started
    service.predict(image)
    legacy_predict(image, config)
    reused_times, legacy_times, maximum_error = [], [], 0.0
    for index in range(repeats):
        results = {}
        # Alternate measurement order to reduce systematic filesystem/thermal bias.
        for mode in ("legacy", "reused") if index % 2 == 0 else ("reused", "legacy"):
            started = perf_counter()
            results[mode] = (
                legacy_predict(image, config)
                if mode == "legacy"
                else service.predict(image).to_dict()
            )
            (legacy_times if mode == "legacy" else reused_times).append(perf_counter() - started)
        first = np.array(list(results["legacy"]["probabilities"].values()))
        second = np.array(list(results["reused"]["probabilities"].values()))
        maximum_error = max(maximum_error, float(np.max(np.abs(first - second))))
        if results["legacy"]["label"] != results["reused"]["label"] or not np.allclose(
            first, second, atol=1e-6, rtol=1e-5
        ):
            raise ValueError("Benchmark predictions disagree; performance results are invalid.")
    started = perf_counter()
    GradCamExplainer(service).explain(image)
    gradcam = perf_counter() - started
    return {
        "baseline_commit": "fe9c3d2b5fa253f00ca0344c02ddf8ba48d440b6",
        "checkpoint_sha256": service.checkpoint_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "threads": threads,
        },
        "measurement": "Both algorithms use the current runtime, decoded image and warm OS cache; imports and decoding excluded. Not a disk-cold startup benchmark.",
        "repeats": repeats,
        "model_startup_ms": startup * 1000,
        "legacy_median_ms": statistics.median(legacy_times) * 1000,
        "reused_median_ms": statistics.median(reused_times) * 1000,
        "speedup": statistics.median(legacy_times) / statistics.median(reused_times),
        "gradcam_ms": gradcam * 1000,
        "model_loads_in_reused_service": service.model_load_count,
        "maximum_probability_error": maximum_error,
        "legacy_samples_ms": [value * 1000 for value in legacy_times],
        "reused_samples_ms": [value * 1000 for value in reused_times],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/best_model.pth"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/benchmark.json")
    args = parser.parse_args()
    report = benchmark(args.checkpoint, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Legacy {report['legacy_median_ms']:.1f} ms; reused {report['reused_median_ms']:.1f} ms; {report['speedup']:.2f}x; output error {report['maximum_probability_error']:.2g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
