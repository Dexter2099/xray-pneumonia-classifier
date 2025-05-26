import os
import torch
from torchvision import transforms
from PIL import Image

from src.model import SimpleCNN


def predict_image(img, config):
    """Load model checkpoint and predict label for a single image.

    Parameters
    ----------
    img : PIL.Image or torch.Tensor
        Input image. If tensor, should be shape (1, H, W) in range [0, 1].
    config : dict
        Configuration dictionary containing at least 'checkpoint_dir'.

    Returns
    -------
    str
        Predicted label: 'NORMAL' or 'PNEUMONIA'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if isinstance(img, torch.Tensor):
        tensor = img
        if tensor.ndim == 3:  # (C,H,W) or (1,H,W)
            pass
        else:
            raise ValueError("Tensor img must have shape (C,H,W)")
        tensor = transforms.Normalize([0.5], [0.5])(tensor)
    else:
        img = img.convert("L")
        tensor = preprocess(img)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)

    return "PNEUMONIA" if pred.item() == 1 else "NORMAL"
