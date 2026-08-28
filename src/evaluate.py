import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import PlantCNN

def evaluate_model():
    data_dir = "data/raw"
    batch_size = 32
    class_names = ['diseased', 'healthy']
    num_classes = len(class_names)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset and get validation split (same seed as training)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(123))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model weights
    device = torch.device("cpu")
    model = PlantCNN(num_classes=num_classes)
    model_path = "outputs/models/plant_cnn.pth"
    
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Run training first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0

    print("\n--- Evaluating Model on Validation Set ---")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Total Validation Images: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Validation Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    evaluate_model()