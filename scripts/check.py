"""Run the repository's verification gates; failures never trigger automatic repairs."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "verification" / "checks.json"


def git_output(arguments):
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def run_check(name, argv, env, cwd):
    display_argv = ["python", *argv[1:]] if argv[0] == sys.executable else argv
    print(f"Running {name}: {' '.join(display_argv)}", flush=True)
    start = time.perf_counter()
    try:
        proc = subprocess.run(argv, env=env, cwd=cwd)
        exit_code = proc.returncode
    except FileNotFoundError:
        exit_code = 127
    except KeyboardInterrupt:
        exit_code = 130
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "command": display_argv,
        "exit_code": exit_code,
        "seconds": round(elapsed, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Engineering verification harness")
    parser.add_argument("--trained", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.release:
        args.trained = True
        args.benchmark = True
        args.audit = True

    if args.benchmark and not args.trained:
        parser.error("--benchmark requires --trained (or use --release)")

    output_path = args.output
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    env = os.environ.copy()
    env.update(
        {"QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    )

    py = sys.executable
    checks_to_run = [
        ("ruff_check", [py, "-m", "ruff", "check", "."]),
        ("ruff_format", [py, "-m", "ruff", "format", "--check", "."]),
        ("mypy", [py, "-m", "mypy"]),
        ("pytest", [py, "-B", "-m", "pytest", "-q", *(["--run-trained"] if args.trained else [])]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]

    if args.trained:
        checks_to_run.extend(
            [
                ("verify_regression", [py, "-B", "-m", "scripts.verify_regression"]),
                ("viewer_smoke", [py, "-B", "-m", "scripts.viewer_smoke"]),
            ]
        )

    if args.benchmark:
        checks_to_run.append(
            ("benchmark_inference", [py, "-B", "-m", "scripts.benchmark_inference"])
        )

    if args.audit:
        checks_to_run.append(
            ("pip_audit", [py, "-m", "pip_audit", "-r", "requirements-viewer.txt"])
        )

    started_utc = datetime.now(UTC).isoformat()
    commit = git_output(["rev-parse", "HEAD"])
    git_status = git_output(["status", "--porcelain"])
    results = []
    not_run = []
    passed = True
    exit_status = 0

    for index, (name, argv) in enumerate(checks_to_run):
        result = run_check(name, argv, env, ROOT)
        results.append(result)
        if result["exit_code"] != 0:
            # Stop at the failed gate and record omissions explicitly; an unrun
            # trained regression or audit must never appear to have passed.
            passed = False
            exit_status = result["exit_code"] if result["exit_code"] > 0 else 1
            not_run = [check[0] for check in checks_to_run[index + 1 :]]
            break

    report = {
        "started_utc": started_utc,
        "commit": commit,
        "working_tree_dirty": None if git_status is None else bool(git_status),
        "flags": {
            "trained": args.trained,
            "benchmark": args.benchmark,
            "audit": args.audit,
            "release": args.release,
        },
        "passed": passed,
        "checks": results,
        "not_run": not_run,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    status_str = "PASS" if passed else "FAIL"
    print(f"{status_str} Report: {output_path}", flush=True)
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
