import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir, batch_size=32, img_size=128):
    """
    Loads images from directory, applies transforms, and splits into train/val loaders.
    """
    # Define image transformations (resize and convert to tensor & normalize)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset from folder
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Directory '{data_dir}' not found. Please add your image class folders inside it.")
        return None, None, []

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    print(f"Found classes: {class_names}")

    # Split dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names

if __name__ == "__main__":
    print("Preprocessing module loaded.")