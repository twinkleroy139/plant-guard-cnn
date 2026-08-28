import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import PlantCNN

# Page Configuration
st.set_page_config(
    page_title="Plant-Guard: Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# App Header
st.title("🌿 Plant-Guard: Crop Disease Detection")
st.markdown("Upload or capture a photo of a plant leaf to instantly detect whether it's **Healthy** or **Diseased**.")

# Load Model Function (Cached for performance)
@st.cache_resource
def load_model():
    class_names = ['diseased', 'healthy']
    num_classes = len(class_names)
    
    model = PlantCNN(num_classes=num_classes)
    model_path = "outputs/models/plant_cnn.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    else:
        st.warning("Trained model weights not found locally. Using uninitialized model for demo structure.")
    
    model.eval()
    return model, class_names

model, class_names = load_model()

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Input Options: File Uploader or Camera
option = st.radio("Choose input method:", ("Upload Leaf Image", "Capture via Camera"))

image = None
if option == "Upload Leaf Image":
    uploaded_file = st.file_uploader("Upload an image file (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    camera_file = st.camera_input("Take a picture of the leaf")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# Display and Predict
if image is not None:
    st.image(image, caption="Selected Leaf Image", use_column_width=True)
    
    if st.button("Diagnose Plant"):
        with st.spinner("Analyzing leaf patterns using CNN..."):
            # Preprocess and predict
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            result = class_names[predicted_class.item()]
            conf_percentage = confidence.item() * 100
            
            # Show Results visually
            st.markdown("---")
            st.subheader("Diagnostic Results")
            if result == "healthy":
                st.success(f"**Diagnosis: HEALTHY** ({conf_percentage:.2f}% Confidence)")
            else:
                st.error(f"**Diagnosis: DISEASED** ({conf_percentage:.2f}% Confidence)")