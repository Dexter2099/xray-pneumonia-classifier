from pathlib import Path

from scripts import record_demo


def test_demo_recorder_paths_and_viewport_are_portfolio_ready():
    assert record_demo.OUTPUT_DIR == record_demo.ROOT / "demo-output"
    assert record_demo.OUTPUT_PATH == record_demo.OUTPUT_DIR / "xray-classifier-demo.mp4"
    assert record_demo.VIEWPORT == {"width": 1440, "height": 900}


def test_find_demo_samples_requires_normal_and_pneumonia_images(tmp_path):
    sample_dir = tmp_path / "demo-assets" / "sample-xrays"
    sample_dir.mkdir(parents=True)

    try:
        record_demo.find_demo_samples(sample_dir)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing sample images to fail clearly")

    assert str(sample_dir) in message
    assert "NORMAL" in message
    assert "PNEUMONIA" in message


def test_find_demo_samples_accepts_labelled_images(tmp_path):
    sample_dir = tmp_path / "demo-assets" / "sample-xrays"
    sample_dir.mkdir(parents=True)
    normal = sample_dir / "normal-example.jpeg"
    pneumonia = sample_dir / "pneumonia-example.jpeg"
    normal.write_bytes(b"normal")
    pneumonia.write_bytes(b"pneumonia")

    samples = record_demo.find_demo_samples(sample_dir)

    assert samples == {"NORMAL": normal, "PNEUMONIA": pneumonia}


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
