import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import yaml

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=224):
        self.image_paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize grayscale images
        ])

        split_dir = os.path.join(root_dir, split)
        for label in ['NORMAL', 'PNEUMONIA']:
            label_dir = os.path.join(split_dir, label)
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter only image files
                    img_path = os.path.join(label_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(0 if label == 'NORMAL' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

def get_dataloaders(config):
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    image_size = config['image_size']
    num_workers = config['num_workers']

    train_dataset = ChestXrayDataset(data_dir, 'train', image_size)
    val_dataset = ChestXrayDataset(data_dir, 'val', image_size)
    test_dataset = ChestXrayDataset(data_dir, 'test', image_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# Sets up the data loading pipeline with transformations and supports train/val/test splits.