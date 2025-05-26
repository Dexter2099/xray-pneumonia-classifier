import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import get_dataloaders
from src.model import SimpleCNN

def evaluate_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(config)

    model = SimpleCNN()
    model.load_state_dict(torch.load(f"{config['checkpoint_dir']}/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, config['visualizations_dir'])

def plot_confusion_matrix(cm, save_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()
#Confusion matrix plot saved in the visualizations directory