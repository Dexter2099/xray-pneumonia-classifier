from pathlib import Path

from scripts import record_demo


def test_demo_recorder_paths_and_viewport_are_portfolio_ready():
    assert record_demo.OUTPUT_DIR == record_demo.ROOT / "demo-output"
    assert record_demo.OUTPUT_PATH == record_demo.OUTPUT_DIR / "xray-classifier-demo.mp4"
    assert record_demo.VIEWPORT == {"width": 1440, "height": 900}


def test_find_demo_samples_requires_metadata_file(tmp_path):
    sample_dir = tmp_path / "demo-assets" / "sample-xrays"
    sample_dir.mkdir(parents=True)

    try:
        record_demo.find_demo_samples(sample_dir)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing sample images to fail clearly")

    assert str(sample_dir / "demo_samples.json") in message
    assert "python scripts/select_demo_samples.py" in message


def test_find_demo_samples_requires_three_case_metadata(tmp_path):
    sample_dir = tmp_path / "demo-assets" / "sample-xrays"
    sample_dir.mkdir(parents=True)
    (sample_dir / "demo_samples.json").write_text(
        """
        [
          {"case_id": "confirmed_pneumonia", "filename": "pneumonia-confirmed.jpeg"},
          {"case_id": "normal", "filename": "normal-example.jpeg"}
        ]
        """,
        encoding="utf-8",
    )

    try:
        record_demo.find_demo_samples(sample_dir)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing demo_samples.json to fail clearly")

    assert "python scripts/select_demo_samples.py" in message
    assert "borderline" in message


def test_find_demo_samples_accepts_metadata_cases(tmp_path):
    sample_dir = tmp_path / "demo-assets" / "sample-xrays"
    sample_dir.mkdir(parents=True)
    filenames = {
        "confirmed_pneumonia": "pneumonia-confirmed.jpeg",
        "normal": "normal-example.jpeg",
        "borderline": "borderline-pneumonia.jpeg",
    }
    for filename in filenames.values():
        (sample_dir / filename).write_bytes(b"image")
    (sample_dir / "demo_samples.json").write_text(
        """
        [
          {"case_id": "confirmed_pneumonia", "filename": "pneumonia-confirmed.jpeg"},
          {"case_id": "normal", "filename": "normal-example.jpeg"},
          {"case_id": "borderline", "filename": "borderline-pneumonia.jpeg"}
        ]
        """,
        encoding="utf-8",
    )

    samples = record_demo.find_demo_samples(sample_dir)

    assert [sample["case_id"] for sample in samples] == [
        "confirmed_pneumonia",
        "normal",
        "borderline",
    ]
    assert [sample["path"] for sample in samples] == [
        sample_dir / filenames["confirmed_pneumonia"],
        sample_dir / filenames["normal"],
        sample_dir / filenames["borderline"],
    ]


def test_build_ffmpeg_command_targets_final_mp4():
    command = record_demo.build_ffmpeg_command(
        Path("raw.webm"),
        Path("demo-output/xray-classifier-demo.mp4"),
        "ffmpeg",
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "raw.webm",
        "-vf",
        "scale=1440:900",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "demo-output/xray-classifier-demo.mp4",
    ]


def test_demo_recorder_has_required_model_explanation_stop():
    assert record_demo.MODEL_EXPLANATION_HEADING == "Model explanation"
    assert callable(record_demo.scroll_to_model_explanation)
