# 🛰️ Poverty Prediction from Satellite Imagery (Kaggle Notebook)

An **unsupervised machine learning** project that predicts poverty levels directly from **satellite images**.  
It uses **deep learning feature extraction (ResNet50)** and **K-Means clustering** — all packaged in a fully interactive **Kaggle Notebook** that needs no web app or external server.

---

## 📖 Project Overview

The project estimates the **economic well-being** of an area (`Poor`, `Middle Class`, or `Rich`) from satellite imagery.  
It follows a clean two-part workflow:

### 🧩 Part 1 — Pre-computation (Training)
- A **ResNet50** model extracts high-dimensional feature vectors from thousands of images.
- **K-Means** clustering groups these vectors into three clusters (representing poverty levels).
- The trained model and extracted features are saved as:
  - `kmeans_model.joblib`
  - `image_features.npy`

### ⚡ Part 2 — Interactive Inference
- The **final notebook (`final_poverty_detection.ipynb`)** loads the trained model.
- A simple **ipywidgets UI** lets users upload new images.
- Upon upload:
  - The image is processed through ResNet50.
  - K-Means predicts its poverty cluster.
  - The notebook displays the prediction and detailed analytics:
    - Poverty level (`Poor`, `Middle Class`, or `Rich`)
    - Image metadata
    - Color histogram
    - Confidence score (based on feature distance to clusters)

---

## ✨ Key Features

| Feature | Description |
|----------|--------------|
| 🧠 **Unsupervised Learning** | No labeled training data needed |
| 💻 **All in Notebook** | Works fully inside Kaggle — no Streamlit, no ngrok |
| ⚙️ **Two-Part Design** | Training and inference separated for speed and clarity |
| 📊 **Rich Analytics** | Confidence score, image histogram, and metadata per prediction |
| 🖱️ **Interactive UI** | Upload and predict using ipywidgets |

---

## ⚙️ How to Run This Project

This project has **two Kaggle notebooks**:
1. `pre-computation-notebook-ipynb.ipynb`
2. `final_poverty_detection.ipynb`

You must **run Part 1 once** to create the model files, then **use Part 2** for interactive predictions.

---

### 🧠 **Part 1: Pre-computation (One-Time Only)**

1. Open **`pre-computation-notebook-ipynb.ipynb`** in **Kaggle**.  
2. Add dataset:  
   `sandeshbhat/satellite-images-to-predict-povertyafrica`
3. Enable **GPU Accelerator** in notebook settings.  
4. Run all cells — this takes about **5–10 minutes**.  
5. The notebook will generate two files in the output directory:
   - `kmeans_model (1).joblib`
   - `image_features.npy`
6. Save the output as a **new private Kaggle Dataset**, e.g. `poverty-prediction`.

---

### 📈 **Part 2: Running the Interactive Prediction Notebook**

1. Open **`final_poverty_detection.ipynb`** in **Kaggle**.  
2. Add your Kaggle dataset (the one saved in Part 1).  
3. Enable **Internet** to download the ResNet50 model.  
4. Run all cells sequentially.  
5. At the end, you’ll see the **Poverty Prediction Interface** with an **Upload Image** button.  
6. Upload a `.png` satellite image and view:
   - Predicted poverty class
   - Image info
   - Color histogram
   - Confidence analysis

---

## 🧭 Notebook Structure (`final_poverty_detection.ipynb`)

| Cell | Purpose |
|------|----------|
| **1** | Installs and imports libraries (`ipywidgets`, `scikit-learn==1.2.2`, `torch`, `matplotlib`, etc.) |
| **2** | Loads `kmeans_model.joblib` and initializes ResNet50 |
| **3** | Defines `predict_image()` → processes image, computes feature distances, and returns predictions |
| **4** | Builds the **ipywidgets UI**, handles uploads, and displays analytics in real-time |

---

## 🧰 Dependencies

```
torch
torchvision
numpy
scikit-learn==1.2.2
opencv-python-headless
matplotlib
ipywidgets
Pillow
```

---

## 🪄 Future Extensions

- Fine-tune ResNet50 using limited labeled samples for better accuracy  
- Add geospatial mapping for regional poverty visualization  
- Deploy via Streamlit or Hugging Face for public demos  

---

**Developed on Kaggle • Powered by PyTorch + Scikit-learn + ipywidgets**
