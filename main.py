import argparse
import yaml
import os
from src.train import train_model
from src.evaluate import evaluate_model
from src.gradcam import visualize_gradcam

def load_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_directories(config):
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['visualizations_dir'], exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chest X-ray Pneumonia Detection')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'gradcam'], required=True, help="Execution mode")
    parser.add_argument('--image', type=str, help="Path to image for Grad-CAM (required for gradcam mode)")

    args = parser.parse_args()
    config = load_config()
    ensure_directories(config)

    print(f"[INFO] Running in {args.mode} mode...")

    if args.mode == 'train':
        print("[INFO] Starting training...")
        train_model(config)
        print("[INFO] Training completed.")

    elif args.mode == 'eval':
        print("[INFO] Starting evaluation...")
        evaluate_model(config)
        print("[INFO] Evaluation completed. Check outputs/visualizations/ for confusion matrix.")

    elif args.mode == 'gradcam':
        if not args.image:
            raise ValueError("You must provide --image path for Grad-CAM mode.")
        print(f"[INFO] Running Grad-CAM on image: {args.image}")
        save_path = os.path.join(config['visualizations_dir'], 'gradcam_result.png')
        visualize_gradcam(config, args.image, save_path)
        print(f"[INFO] Grad-CAM saved to {save_path}")
