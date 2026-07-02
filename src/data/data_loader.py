# Module: Data Loading and Preprocessing

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

def map_and_validate_dataset(base_dir: str, image_type: str = 'Original_images') -> tuple:
    """Navigates the specific AneRBC nested structure to map images to labels."""
    image_paths = []
    labels = []
    base_path = Path(base_dir)
    datasets = ['AneRBC-I', 'AneRBC-II']
    classes = {'Anemic_individuals': 'Anemic', 'Healthy_individuals': 'Healthy'}
    
    for ds in datasets:
        for class_folder, label_name in classes.items():
            target_path = base_path / ds / class_folder / image_type
            if target_path.exists() and target_path.is_dir():
                for img_file in target_path.glob('*.png'):
                    image_paths.append(str(img_file))
                    labels.append(label_name)
    return image_paths, labels

def create_stratified_subset(image_paths: list, labels: list, total_samples: int = 2000, seed: int = 42) -> tuple:
    """Reduces the dataset size for faster training while maintaining class balance."""
    if total_samples >= len(image_paths):
        return image_paths, labels
        
    subset_paths, _, subset_labels, _ = train_test_split(
        image_paths, labels, 
        train_size=total_samples, 
        stratify=labels, 
        random_state=seed
    )
    return subset_paths, subset_labels

class AneRBCDataset(Dataset):
    """Custom PyTorch Dataset for loading and transforming AneRBC images."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def prepare_dataloaders(image_paths: list, labels: list, batch_size: int = 32, seed: int = 42) -> tuple:
    """Splits data deterministically and creates PyTorch DataLoaders."""
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, encoded_labels, test_size=0.30, stratify=encoded_labels, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AneRBCDataset(X_train, y_train, transform=transform_pipeline)
    val_dataset = AneRBCDataset(X_val, y_val, transform=transform_pipeline)
    test_dataset = AneRBCDataset(X_test, y_test, transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, le