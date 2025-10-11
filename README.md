Poverty Level Prediction from Satellite Imagery
An unsupervised machine learning project that uses deep learning features to classify satellite images into different poverty levels. The project includes an interactive web application built with Streamlit for real-time predictions.

Project Overview
This project explores the potential of using publicly available satellite imagery to identify economic well-being in a region. By applying a combination of transfer learning and unsupervised clustering, the model can group images into distinct categories corresponding to 'Poor', 'Middle Class', and 'Rich' areas without requiring pre-labeled data.

The core methodology involves:

Feature Extraction: Using a pre-trained ResNet50 model to convert each satellite image into a high-dimensional feature vector.

Clustering: Applying the K-Means algorithm to group these feature vectors into three distinct clusters.

Prediction: Building an interactive web application with Streamlit that allows a user to upload a new satellite image and see its predicted poverty level in real-time.

Features
Deep Feature Extraction: Leverages the power of a state-of-the-art computer vision model (ResNet50).

Unsupervised Learning: Does not require manually labeled training data, making it highly scalable.

Interactive Web UI: A user-friendly Streamlit interface to upload images and view predictions instantly.

Kaggle Deployment: Includes a setup to run the entire application within a Kaggle Notebook, exposed to the web via ngrok.

Technologies Used
Backend & Modeling: Python, PyTorch, Torchvision, scikit-learn, OpenCV

Frontend: Streamlit

Environment: Kaggle Notebooks, Jupyter

Deployment Tunneling: pyngrok

Setup and Installation
There are two primary ways to run this project: on your local machine or directly within a Kaggle Notebook.

1. Running on a Kaggle Notebook (Recommended)
This is the easiest way to get started, as the environment and dataset are managed by Kaggle.

Step 1: Setup the Notebook

Create a new Kaggle Notebook.

Add the dataset by clicking "+ Add data" and searching for sandeshbhat/satellite-images-to-predict-povertyafrica.

Enable Internet in the notebook's settings panel on the right.

Add your secrets using the "Add-ons" > "Secrets" menu:

KAGGLE_USERNAME: Your Kaggle username.

KAGGLE_KEY: Your Kaggle API key from the kaggle.json file.

NGROK_AUTHTOKEN: Your authentication token from the ngrok dashboard.

Step 2: Run the Code

The project's Kaggle Notebook contains two cells.

Run the first cell to write the Streamlit application code to a file named app.py.

Run the second cell to install the required libraries and launch the Streamlit app via an ngrok tunnel.

Click the public .ngrok.io URL printed in the output to access your live application.

2. Running on a Local Machine
Step 1: Clone the Repository

Bash

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Step 2: Create a Virtual Environment (Recommended)

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Step 3: Install Dependencies

Bash

pip install -r requirements.txt
(You will need to create a requirements.txt file containing streamlit, torch, torchvision, scikit-learn, opencv-python, kaggle, and pyngrok).

Step 4: Download the Dataset

Set up the Kaggle API by placing your kaggle.json file in the appropriate directory (e.g., ~/.kaggle/ on Linux/macOS).

Download the dataset:

Bash

kaggle datasets download -d sandeshbhat/satellite-images-to-predict-povertyafrica -p data/ --unzip
You will need to modify the image_dir path in app.py to point to data/nigeria_archive/images.

Step 5: Run the Streamlit App

Bash

streamlit run app.py
Usage
Once the application is running, open the provided URL in your browser.

Click the "Browse files" button to upload a .png satellite image.

The application will process the image and display the predicted poverty level ('Poor', 'Middle Class', or 'Rich').

Future Improvements
Supervised Learning: Fine-tune the ResNet50 model using a small set of labeled images to potentially improve accuracy.

Explore Other Models: Experiment with different pre-trained models (e.g., EfficientNet, Vision Transformer) for feature extraction.

Persistent Deployment: Deploy the application on a permanent cloud service like Streamlit Community Cloud, Heroku, or AWS.
