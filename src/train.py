import os
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_data_loaders
from model import PlantCNN

def train_model():
    data_dir = "data/raw"
    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    # Load data
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    if not class_names:
        print("Please place your class image folders inside 'data/raw/' before training.")
        return

    num_classes = len(class_names)
    print(f"Training model for {num_classes} classes: {class_names}")

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PlantCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Save trained model weights
    os.makedirs("outputs/models", exist_ok=True)
    model_path = "outputs/models/plant_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()