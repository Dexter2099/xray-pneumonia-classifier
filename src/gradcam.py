"""Grad-CAM with scoped hooks and a reusable prediction model."""

from collections.abc import Mapping
from typing import Protocol

import cv2
import numpy as np
import torch
from PIL import Image

from src.errors import ExplanationError
from src.inference import PredictionService
from src.inputs import ImageSource, load_image, preprocess_image
from src.settings import InferenceSettings


def normalize_cam(cam: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not cam.size or not np.isfinite(cam).all():
        raise ExplanationError("Explanation contains empty or non-finite activations.")
    cam = cam - np.min(cam)
    maximum = float(np.max(cam))
    if maximum <= epsilon:
        # A constant map contains no localisation information. Returning zeros lets
        # the explainer report that honestly instead of dividing by zero or inventing contrast.
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / maximum).astype(np.float32)


class GradCAM:
    """Each generation owns its hooks, graph and intermediate tensors."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ExplanationError("Grad-CAM requires exactly one batched image.")

        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        handles = []
        try:
            handles.append(self.target_layer.register_forward_hook(forward_hook))
            handles.append(self.target_layer.register_full_backward_hook(backward_hook))
            # eval() disables dropout but does not disable autograd. Override any
            # enclosing inference context for this separate explanation pass only.
            with torch.inference_mode(False), torch.enable_grad():
                output = self.model(input_tensor.detach().clone())
                index = int(output[0].argmax().item()) if class_idx is None else class_idx
                if not 0 <= index < output.shape[1]:
                    raise ExplanationError(
                        "Explanation class index is outside the model's classes."
                    )
                self.model.zero_grad(set_to_none=True)
                output[0, index].backward()
                if self.gradients is None or self.activations is None:
                    raise ExplanationError(
                        "The target layer did not produce explanation gradients."
                    )
                gradients = self.gradients[0].detach().cpu().numpy()
                activations = self.activations[0].detach().cpu().numpy()
                weights = gradients.mean(axis=(1, 2))
                # Spatially averaged gradients weight each channel's influence on
                # the selected logit; ReLU retains positive contributions only.
                cam = np.maximum((weights[:, None, None] * activations).sum(axis=0), 0)
                return normalize_cam(cv2.resize(cam, (224, 224)))
        except RuntimeError as exc:
            raise ExplanationError("Grad-CAM could not compute gradients for this image.") from exc
        finally:
            # Hooks otherwise survive on the shared model and retain later graphs.
            # Failure paths need the same cleanup as successful explanations.
            for handle in handles:
                handle.remove()
            self.gradients = None
            self.activations = None
            self.model.zero_grad(set_to_none=True)


class Explainer(Protocol):
    """Return an RGB explanation or an actionable XrayError; prediction is independent."""

    def explain(self, image: ImageSource) -> Image.Image: ...


class GradCamExplainer:
    def __init__(self, service: PredictionService) -> None:
        self.service = service

    def explain(self, image: ImageSource) -> Image.Image:
        decoded = load_image(image)
        tensor = preprocess_image(decoded).unsqueeze(0).to(self.service.device)
        with self.service.lock:
            cam = GradCAM(self.service.model, self.service.model.conv3).generate(tensor)
        if not np.any(cam > 0):
            raise ExplanationError("No usable heatmap signal. The prediction is still available.")
        grayscale = np.asarray(decoded.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # OpenCV uses BGR while PIL/Qt expect RGB; blend in one colour space, then convert.
        overlay = cv2.addWeighted(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def generate_gradcam_overlay(
    image: ImageSource, config: Mapping, *, service: PredictionService | None = None
) -> Image.Image:
    engine = service or PredictionService(InferenceSettings.from_mapping(config))
    return GradCamExplainer(engine).explain(image)


def visualize_gradcam(config: Mapping, image_path, save_path) -> None:
    import matplotlib.pyplot as plt

    overlay = generate_gradcam_overlay(image_path, config)
    figure, axis = plt.subplots()
    try:
        axis.imshow(overlay)
        axis.set_title("Grad-CAM")
        axis.axis("off")
        figure.savefig(save_path)
    finally:
        plt.close(figure)
