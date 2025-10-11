# 🌍 Poverty Level Prediction from Satellite Imagery

An **unsupervised machine learning** project that uses deep learning features to classify satellite images into different poverty levels.  
Includes an **interactive Streamlit web app** for real-time predictions.

---

## 🚀 Project Overview

This project explores how publicly available **satellite imagery** can help identify economic well-being in a region.  
By combining **transfer learning** and **unsupervised clustering**, the model groups images into categories — **Poor**, **Middle Class**, and **Rich** — without pre-labeled data.

### Core Methodology
1. **Feature Extraction:**  
   Uses a pre-trained **ResNet50** model to convert each image into a high-dimensional feature vector.
2. **Clustering:**  
   Applies **K-Means** to group feature vectors into three clusters.
3. **Prediction:**  
   An **interactive Streamlit app** allows users to upload new satellite images and view predicted poverty levels in real time.

---

## ✨ Features

- **Deep Feature Extraction:** Uses state-of-the-art computer vision models (ResNet50).  
- **Unsupervised Learning:** No need for labeled training data.  
- **Interactive Web UI:** Streamlit interface for instant predictions.  
- **Kaggle Deployment:** Run the full pipeline inside Kaggle with **ngrok** exposure.

---

## 🧠 Technologies Used

| Category | Tools |
|-----------|-------|
| **Backend & Modeling** | Python, PyTorch, Torchvision, scikit-learn, OpenCV |
| **Frontend** | Streamlit |
| **Environment** | Kaggle Notebooks, Jupyter |
| **Deployment Tunneling** | pyngrok |

---

## ⚙️ Setup and Installation

There are two main ways to run this project:

### **1️⃣ Running on Kaggle (Recommended)**

#### Step 1: Setup the Notebook
- Create a new **Kaggle Notebook**.  
- Add dataset: `sandeshbhat/satellite-images-to-predict-povertyafrica`.  
- Enable **Internet** in notebook settings.  
- Add secrets under **Add-ons → Secrets**:
  - `KAGGLE_USERNAME`
  - `KAGGLE_KEY`
  - `NGROK_AUTHTOKEN`

#### Step 2: Run the Code
- The notebook has two main cells:
  1. Writes `app.py` (the Streamlit app).
  2. Installs dependencies and runs the app via ngrok.
- After execution, click the **public `.ngrok.io` URL** to view your app live.

---

### **2️⃣ Running on Local Machine**

#### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

#### Step 2: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

`requirements.txt` should include:
```
streamlit
torch
torchvision
scikit-learn
opencv-python
kaggle
pyngrok
```

#### Step 4: Download the Dataset
```bash
kaggle datasets download -d sandeshbhat/satellite-images-to-predict-povertyafrica -p data/ --unzip
```
> Make sure `app.py` points to `data/nigeria_archive/images`.

#### Step 5: Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app URL in your browser.  
2. Upload a `.png` satellite image.  
3. View the predicted poverty level:
   - **Poor**
   - **Middle Class**
   - **Rich**

---

## 🔮 Future Improvements

- **Supervised Learning:** Fine-tune ResNet50 with labeled data.  
- **Model Exploration:** Try models like EfficientNet or Vision Transformer.  
- **Persistent Deployment:** Host on Streamlit Cloud, Heroku, or AWS for long-term access.

---

**Developed with ❤️ using PyTorch and Streamlit**
