import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from src.model import SimpleCNN

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

def visualize_gradcam(config, image_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(f"{config['checkpoint_dir']}/best_model.pth", map_location=device))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img = Image.open(image_path).convert('L')
    input_tensor = transform(img).unsqueeze(0).to(device)

    gradcam = GradCAM(model, model.conv3)
    cam = gradcam.generate(input_tensor)

    img = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
# GradCAM class for computing saliency maps
# visualize_gradcam() function to apply Grad-CAM to an input X-ray image and save the visualization