"""Dispatch training, final evaluation and Grad-CAM with the same configuration policy."""

import argparse

from src.settings import InferenceSettings, load_config, resolve_project_path


def ensure_directories(config):
    # Match inference's repository-relative policy even when launched from elsewhere.
    InferenceSettings.from_mapping(config).checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    resolve_project_path(config["visualizations_dir"]).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chest X-ray Pneumonia Detection")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "gradcam"],
        required=True,
        help="Execution mode",
    )
    parser.add_argument(
        "--image", type=str, help="Path to image for Grad-CAM (required for gradcam mode)"
    )

    args = parser.parse_args()
    config = load_config()
    ensure_directories(config)

    print(f"[INFO] Running in {args.mode} mode...")

    if args.mode == "train":
        from src.train import train_model

        print("[INFO] Starting training...")
        train_model(config)
        print("[INFO] Training completed.")

    elif args.mode == "eval":
        from src.evaluate import evaluate_model

        print("[INFO] Starting evaluation...")
        evaluate_model(config)
        print("[INFO] Evaluation completed. Check outputs/visualizations/ for confusion matrix.")

    elif args.mode == "gradcam":
        from src.gradcam import visualize_gradcam

        if not args.image:
            raise ValueError("You must provide --image path for Grad-CAM mode.")
        print(f"[INFO] Running Grad-CAM on image: {args.image}")
        save_path = resolve_project_path(config["visualizations_dir"]) / "gradcam_result.png"
        visualize_gradcam(config, args.image, save_path)
        print(f"[INFO] Grad-CAM saved to {save_path}")
