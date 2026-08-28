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

# Load External Stylesheet
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("static/styles.css")

# App Header
st.title("🌿 Plant-Guard")
st.markdown("### AI-Powered Crop Disease Diagnostic System")
st.write("Upload a clear photo of a plant leaf or capture one directly using your camera to receive an instant CNN classification.")

# Load Model (Cached)
@st.cache_resource
def load_model():
    class_names = ['diseased', 'healthy']
    num_classes = len(class_names)
    
    model = PlantCNN(num_classes=num_classes)
    model_path = "outputs/models/plant_cnn.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    model.eval()
    return model, class_names

model, class_names = load_model()

# Image Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Input Interface Selection
input_method = st.radio("Select Input Method:", ("Upload Image File", "Capture via Camera"))

image = None
if input_method == "Upload Image File":
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    camera_file = st.camera_input("Position leaf in front of camera")
    if camera_file is not None:
        image = Image.open(camera_file).convert('RGB')

# Prediction Section
if image is not None:
    st.image(image, caption="Analyzed Leaf Sample", use_container_width=True)
    
    if st.button("Run Diagnostic Analysis"):
        with st.spinner("Processing through Convolutional Neural Network..."):
            input_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            result = class_names[predicted_class.item()]
            conf_percentage = confidence.item() * 100
            
            st.markdown("---")
            if result == "healthy":
                st.success(f"### Result: HEALTHY Leaf\n**Confidence Score:** {conf_percentage:.2f}%")
            else:
                st.error(f"### Result: DISEASED Leaf\n**Confidence Score:** {conf_percentage:.2f}%")