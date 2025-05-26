import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.model import SimpleCNN
from src.dataset import get_dataloaders

def train_model(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/best_model.pth")

    print("Training complete.")

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total
# Script covers the training loop with validation accuracy monitoring and checkpoint saving.