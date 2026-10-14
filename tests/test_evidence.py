import copy
import json

import pytest

from src.settings import ROOT
from src.viewer.evidence import MAX_REPORT_BYTES, read_evidence


@pytest.fixture
def report(tmp_path):
    value = json.loads((ROOT / "docs/evidence/regression.json").read_text())
    value.update(
        source_revision="a" * 40, working_tree_dirty=False, generated_utc="2026-10-15T00:00:00Z"
    )
    path = tmp_path / "regression.json"
    return path, value


def read(report, *, revision="a" * 40, dirty=False, checkpoint=None):
    path, value = report
    path.write_text(json.dumps(value))
    return read_evidence(path, checkpoint or value["checkpoint_sha256"], revision, dirty)


def test_saved_evidence_recomputes_errors_and_identifies_current_checkout(report):
    evidence = read(report)
    assert evidence.status == "Recorded PASS"
    assert evidence.provenance == "Current clean checkout"
    assert len(evidence.cases) == 3
    assert all(case.passed for case in evidence.cases)
    assert evidence.cases[2].error > 0
    assert "refreshing does not rerun tests" in evidence.detail


@pytest.mark.parametrize(
    "revision,dirty", [("b" * 40, False), (None, False), ("a" * 40, True), ("a" * 40, None)]
)
def test_other_unknown_or_dirty_code_cannot_appear_current(report, revision, dirty):
    assert read(report, revision=revision, dirty=dirty).provenance.startswith("Historical")


def test_old_or_dirty_report_cannot_appear_current(report):
    report[1].pop("source_revision")
    assert read(report).provenance.startswith("Historical")
    report[1].update(source_revision="a" * 40, working_tree_dirty=True)
    assert read(report).provenance.startswith("Historical")


def test_different_active_checkpoint_is_explicit(report):
    assert read(report, checkpoint="0" * 64).provenance == "Different active checkpoint"


@pytest.mark.parametrize(
    "mutation",
    ["status", "expected", "tolerance", "duplicate", "nan", "label", "boolean", "missing"],
)
def test_malformed_or_inconsistent_pass_report_is_rejected(report, mutation):
    value = report[1]
    if mutation == "status":
        value["cases"][0]["actual"]["probabilities"] = {"NORMAL": 0.1, "PNEUMONIA": 0.9}
    elif mutation == "expected":
        value["cases"][0]["expected"]["confidence"] = 0.9
    elif mutation == "tolerance":
        value["absolute_tolerance"] = 1
    elif mutation == "duplicate":
        value["cases"][1] = copy.deepcopy(value["cases"][0])
    elif mutation == "nan":
        value["cases"][0]["actual"]["probabilities"]["NORMAL"] = float("nan")
    elif mutation == "label":
        value["cases"][0]["actual"]["label"] = "NORMAL"
    elif mutation == "boolean":
        value["passed"] = "true"
    else:
        value.pop("cases")
    assert read(report).status == "Unavailable"


def test_real_recorded_failure_is_displayed_as_failure(report):
    value = report[1]
    value["cases"][0]["actual"]["probabilities"] = {"NORMAL": 0.1, "PNEUMONIA": 0.9}
    value["cases"][0]["passed"] = False
    value["passed"] = False
    evidence = read(report)
    assert evidence.status == "Recorded FAIL"
    assert not evidence.cases[0].passed


@pytest.mark.parametrize(
    "content",
    [b"{bad JSON", b"[]", b"x" * (MAX_REPORT_BYTES + 1)],
    ids=["malformed", "array", "oversized"],
)
def test_unreadable_json_is_recoverable(tmp_path, content):
    path = tmp_path / "report.json"
    path.write_bytes(content)
    assert read_evidence(path, "", None, None).status == "Unavailable"


def test_missing_report_is_recoverable(tmp_path):
    assert read_evidence(tmp_path / "missing.json", "", None, None).status == "Unavailable"
