# Plant Guard CNN 🌿🤖

A custom Convolutional Neural Network (CNN) application designed to classify plant leaves as healthy or diseased, featuring enriched agricultural care insights and deployed via Streamlit.

## 🚀 Live Demo
Access the live web application here:
[👉 Click here to view Plant Guard Live](https://plant-guard-cnn.onrender.com)

---

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Frontend UI:** Streamlit, Custom CSS
* **Deployment:** Render Cloud Platform

---

## 🧠 How to Retrain the Model with New Data

To increase model accuracy and improve classification performance as you collect more plant leaf samples, follow these steps locally:

1. **Add Training Images:** 
   Place your new images into their respective folders within your local directory structure:
   * Healthy images go to: `data/raw/healthy/`
   * Diseased images go to: `data/raw/diseased/`

2. **Run the Preprocessing & Training Scripts:**
   Open your terminal in the project root directory and execute your training script:
   ```bash
   python src/preprocess.py
   python src/train.py