"""Read bounded regression evidence without running code or trusting a PASS flag."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.provenance import source_state
from src.settings import ROOT

MAX_REPORT_BYTES = 128 * 1024


@dataclass(frozen=True)
class EvidenceCase:
    name: str
    expected_label: str
    actual_label: str
    expected: tuple[float, float]
    actual: tuple[float, float]
    error: float
    passed: bool


@dataclass(frozen=True)
class VerificationEvidence:
    status: str
    provenance: str
    detail: str
    cases: tuple[EvidenceCase, ...] = ()


def _read(path: Path) -> dict:
    # Bound the read itself, not just a prior stat: the file may grow between calls.
    with path.open("rb") as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the 128 KiB limit.")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Report must contain a JSON object.")
    return value


def _probabilities(prediction: dict) -> tuple[float, float]:
    probabilities = prediction["probabilities"]
    values = tuple(probabilities[label] for label in ("NORMAL", "PNEUMONIA"))
    if any(type(value) not in (float, int) for value in values):
        raise ValueError("Probabilities must be numbers.")
    normal, pneumonia = (float(value) for value in values)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (normal, pneumonia)):
        raise ValueError("Probabilities must be finite and in [0,1].")
    if not math.isclose(normal + pneumonia, 1, abs_tol=1e-5):
        raise ValueError("Probabilities must sum to one.")
    if prediction["label"] != ("NORMAL" if normal >= pneumonia else "PNEUMONIA"):
        raise ValueError("Prediction label disagrees with its probabilities.")
    return normal, pneumonia


def _validate_comparisons(report: dict, baseline: dict) -> tuple[tuple[EvidenceCase, ...], bool]:
    """Pure validation: a saved PASS is accepted only when the frozen comparisons agree."""
    if report["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise ValueError("Evidence checkpoint differs from the frozen baseline.")
    # The saved run cannot choose more permissive tolerances or new expected outputs.
    if any(
        report[key] != baseline[key]
        for key in ("baseline_commit", "absolute_tolerance", "relative_tolerance")
    ):
        raise ValueError("Evidence reference or tolerances differ from the frozen baseline.")
    references = {case["image"]: case for case in baseline["cases"]}
    saved = report["cases"]
    if not isinstance(saved, list) or len(saved) != len(references):
        raise ValueError("Evidence must cover every baseline image exactly once.")
    rows, seen = [], set()
    for case in saved:
        name = case["image"]
        if name in seen or name not in references:
            raise ValueError("Evidence has duplicate or unknown baseline images.")
        seen.add(name)
        expected = references[name]["prediction"]
        if case["expected"] != expected:
            raise ValueError("Saved expectations differ from the frozen baseline.")
        reference, actual = _probabilities(expected), _probabilities(case["actual"])
        agrees = expected["label"] == case["actual"]["label"] and all(
            abs(value - target)
            <= baseline["absolute_tolerance"] + baseline["relative_tolerance"] * abs(target)
            for value, target in zip(actual, reference, strict=True)
        )
        if type(case["passed"]) is not bool or case["passed"] != agrees:
            raise ValueError("Saved case status disagrees with its numerical comparison.")
        rows.append(
            EvidenceCase(
                Path(name).name,
                expected["label"],
                case["actual"]["label"],
                reference,
                actual,
                max(abs(a - b) for a, b in zip(actual, reference, strict=True)),
                agrees,
            )
        )
    batch = report["single_batch_agreement"]
    if type(batch) is not bool or type(report["passed"]) is not bool:
        raise ValueError("Saved statuses must be booleans.")
    passed = all(row.passed for row in rows) and batch
    if report["passed"] != passed:
        raise ValueError("Saved overall status disagrees with the cases and batch check.")
    return tuple(rows), passed


def _describe_evidence(
    report: dict,
    baseline: dict,
    rows: tuple[EvidenceCase, ...],
    passed: bool,
    report_path: Path,
    checkpoint_sha256: str,
    revision: str | None,
    dirty: bool | None,
) -> VerificationEvidence:
    """Label source/checkpoint identity separately from the recorded numerical outcome."""
    recorded_revision = report.get("source_revision")
    current = (
        bool(revision)
        and recorded_revision == revision
        and dirty is False
        and report.get("working_tree_dirty") is False
    )
    provenance = "Current clean checkout" if current else "Historical / unverified source state"
    checkpoint_matches = checkpoint_sha256 == report["checkpoint_sha256"]
    checkpoint_status = "matches" if checkpoint_matches else "DOES NOT MATCH"
    if not checkpoint_sha256:
        checkpoint_status = "unavailable (no loaded model)"
    detail = (
        f"Saved report: {report_path.name}\n"
        f"Recorded code: {recorded_revision or 'not recorded'}\n"
        f"Checkout code: {revision or 'unavailable'}; dirty: {dirty}\n"
        f"Generated UTC: {report.get('generated_utc', 'not recorded')}\n"
        f"Checkpoint SHA-256: {report['checkpoint_sha256']}\n"
        f"Active checkpoint: {checkpoint_status}\n"
        f"Tolerance: abs {baseline['absolute_tolerance']:g}, rel {baseline['relative_tolerance']:g}\n"
        f"Single/batch agreement recorded: {report['single_batch_agreement']}\n"
        "Source/checkpoint state checked at refresh time.\n"
        "Saved evidence only; refreshing does not rerun tests or verify the selected image."
    )
    status = "Recorded PASS" if passed else "Recorded FAIL"
    if not checkpoint_matches:
        provenance = "Different active checkpoint"
    return VerificationEvidence(status, provenance, detail, tuple(rows))


def read_evidence(
    report_path: Path,
    checkpoint_sha256: str,
    revision: str | None,
    dirty: bool | None,
    baseline_path: Path = ROOT / "tests/fixtures/trained_baseline.json",
) -> VerificationEvidence:
    """Recheck saved comparisons against the frozen baseline; this is not a new model run."""
    try:
        report, baseline = _read(report_path), _read(baseline_path)
        rows, passed = _validate_comparisons(report, baseline)
        return _describe_evidence(
            report, baseline, rows, passed, report_path, checkpoint_sha256, revision, dirty
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        return VerificationEvidence("Unavailable", "No usable evidence", str(exc))


def load_verification(checkpoint_sha256: str) -> VerificationEvidence:
    revision, dirty = source_state(ROOT)
    local = ROOT / "outputs/reports/regression.json"
    # A malformed local report stays visible as an error rather than falling back to a PASS.
    path = local if local.exists() else ROOT / "docs/evidence/regression.json"
    return read_evidence(path, checkpoint_sha256, revision, dirty)
