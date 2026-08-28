import torch
from torchvision import transforms
from PIL import Image
from model import PlantCNN

def predict_image(image_path):
    class_names = ['diseased', 'healthy']
    num_classes = len(class_names)
    
    # Load model structure and weights
    device = torch.device("cpu")
    model = PlantCNN(num_classes=num_classes)
    model.load_state_dict(torch.load("outputs/models/plant_cnn.pth", map_location=device))
    model.eval()

    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    result = class_names[predicted_class.item()]
    conf_percentage = confidence.item() * 100

    print(f"\n--- Prediction Result ---")
    print(f"Image: {image_path}")
    print(f"Diagnosis: {result.upper()} ({conf_percentage:.2f}% confidence)")

if __name__ == "__main__":
    # Test with one of your images or any new leaf photo path
    test_img = "data/raw/healthy/4456202-mango-leaves-2846722_1920.jpg"
    predict_image(test_img)