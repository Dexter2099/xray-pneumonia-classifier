import pytest

from src.reporting import build_evaluation_report


def test_evaluation_report_keeps_class_order_and_single_class_splits():
    report = build_evaluation_report([0, 0], [0, 1], "abc", {"image_size": 224})
    assert report["confusion_matrix"] == [[1, 1], [0, 0]]
    assert report["dataset_counts"] == {"NORMAL": 2, "PNEUMONIA": 0}
    assert report["classification_report"]["NORMAL"]["recall"] == 0.5
    assert report["classification_report"]["PNEUMONIA"]["support"] == 0
    assert report["checkpoint_sha256"] == "abc"


@pytest.mark.parametrize("truth,predictions", [([], []), ([0], []), ([2], [0])])
def test_invalid_evaluation_is_not_a_successful_report(truth, predictions):
    with pytest.raises(ValueError):
        build_evaluation_report(truth, predictions, "abc", {})
