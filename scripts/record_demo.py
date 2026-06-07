from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "demo-output"
OUTPUT_PATH = OUTPUT_DIR / "xray-classifier-demo.mp4"
RAW_VIDEO_DIR = OUTPUT_DIR / "raw"
SAMPLE_DIR = ROOT / "demo-assets" / "sample-xrays"
CHECKPOINT_PATH = ROOT / "outputs" / "checkpoints" / "best_model.pth"
VIEWPORT = {"width": 1440, "height": 900}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MODEL_EXPLANATION_HEADING = "Model explanation"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def find_demo_samples(sample_dir: Path = SAMPLE_DIR) -> dict[str, Path]:
    samples: dict[str, Path] = {}
    if sample_dir.exists():
        for path in sorted(sample_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_text = str(path.relative_to(sample_dir)).upper()
            if "NORMAL" in label_text and "NORMAL" not in samples:
                samples["NORMAL"] = path
            elif "PNEUMONIA" in label_text and "PNEUMONIA" not in samples:
                samples["PNEUMONIA"] = path

    missing = [label for label in ("NORMAL", "PNEUMONIA") if label not in samples]
    if missing:
        raise RuntimeError(
            "Missing demo sample image(s) for "
            f"{', '.join(missing)} in {sample_dir}. "
            "Add .png, .jpg, or .jpeg files with NORMAL and PNEUMONIA in the filename or parent folder."
        )
    return samples


def validate_demo_inputs() -> None:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Missing trained checkpoint: {CHECKPOINT_PATH}")
    find_demo_samples()


def wait_for_streamlit(base_url: str, timeout_seconds: int = 45) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url}/_stcore/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not become ready at {base_url}")


def launch_streamlit(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["XRAY_DEMO_RECORDING"] = "1"
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def clear_previous_output(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def find_ffmpeg() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or run `pip install imageio-ffmpeg`."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_ffmpeg_command(input_video: Path, output_video: Path, ffmpeg_executable: str) -> list[str]:
    return [
        ffmpeg_executable,
        "-y",
        "-i",
        input_video.as_posix(),
        "-vf",
        "scale=1440:900",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video.as_posix(),
    ]


def convert_to_mp4(raw_video_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(raw_video_path, output_path, find_ffmpeg())
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def scroll_to_model_explanation(page) -> None:
    heading = page.get_by_role("heading", name=MODEL_EXPLANATION_HEADING)
    heading.wait_for(timeout=15_000)
    heading.scroll_into_view_if_needed(timeout=10_000)
    page.evaluate(
        """
        (headingText) => {
            const headings = Array.from(document.querySelectorAll('h1, h2, h3'));
            const heading = headings.find((node) => node.textContent.trim() === headingText);
            if (!heading) {
                throw new Error(`Heading not found: ${headingText}`);
            }
            const scrollers = Array.from(document.querySelectorAll('*')).filter((node) => {
                const style = window.getComputedStyle(node);
                return /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight;
            });
            const scrollParent = scrollers.find((node) => node.contains(heading));
            if (scrollParent) {
                const currentTop = heading.getBoundingClientRect().top;
                scrollParent.scrollTop += currentTop - 96;
            } else {
                const targetY = Math.max(heading.getBoundingClientRect().top + window.scrollY - 96, 0);
                window.scrollTo({ top: targetY, behavior: 'instant' });
            }
        }
        """,
        MODEL_EXPLANATION_HEADING,
    )
    page.wait_for_timeout(4_500)


def run_demo_flow(base_url: str, video_dir: Path) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements.txt` "
            "and `python -m playwright install chromium`, then retry."
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()
        page.goto(base_url, wait_until="networkidle")
        page.get_by_text("Chest X-Ray Pneumonia Classifier").wait_for(timeout=15_000)
        page.wait_for_timeout(2_500)

        page.get_by_role("button", name="Pneumonia example").click()
        page.wait_for_timeout(1_000)
        page.get_by_role("button", name="Run prediction").click()
        page.locator('[data-testid="stMetric"]').first.wait_for(timeout=45_000)
        page.wait_for_timeout(2_000)

        scroll_to_model_explanation(page)
        evaluation_heading = page.get_by_text("Evaluation artifacts")
        if evaluation_heading.count() > 0:
            evaluation_heading.scroll_into_view_if_needed(timeout=10_000)
            page.wait_for_timeout(2_000)
        page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
        page.wait_for_timeout(2_000)

        video = page.video
        context.close()
        browser.close()
        if video is None:
            raise RuntimeError("Playwright did not create a video artifact.")
        return Path(video.path())


def record_demo() -> Path:
    validate_demo_inputs()
    clear_previous_output()
    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = launch_streamlit(port)
    try:
        wait_for_streamlit(base_url)
        with tempfile.TemporaryDirectory(prefix="xray-classifier-demo-") as temp_dir:
            raw_video_path = run_demo_flow(base_url, Path(temp_dir))
            convert_to_mp4(raw_video_path, OUTPUT_PATH)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return OUTPUT_PATH


def main() -> int:
    try:
        output_path = record_demo()
    except Exception as exc:
        print(f"Recording failed: {exc}", file=sys.stderr)
        return 1

    print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
