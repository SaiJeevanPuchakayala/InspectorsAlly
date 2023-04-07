import streamlit as st
import io
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG


from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")


st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Leather Product Images."
    )


# Define the functions to load images
def load_uploaded_image(file):
    img = Image.open(file)
    return img


# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_folder = "./data/"
subset_name = "leather"
data_folder = os.path.join(data_folder, subset_name)


def Anomaly_Detection(image_path, root):
    """
    Given an image path and a trained PyTorch model, returns the predicted class and bounding boxes for any defects detected in the image.
    """

    batch_size = 1
    threshold = 0.5

    subset_name = "leather"
    model_path = f"./weights/{subset_name}_model.h5"
    model = torch.load(model_path, map_location=device)

    # Get the list of class names from the test loader
    _, test_loader = get_train_test_loaders(root, batch_size=batch_size)
    class_names = test_loader.dataset.classes

    # Load the image and preprocess it
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    image = transform(image_path).unsqueeze(0)

    # Get the model's predictions for the image
    with torch.no_grad():
        output = model(image)
    predicted_probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

    # Get the predicted class label and probability
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = class_names[predicted_class_index]
    prediction_sentence = "Congratulations! Your product has been classified as a 'Good' item with no anomalies detected in the inspection images."
    if predicted_class != "Good":
        prediction_sentence = "We're sorry to inform you that our AI-based visual inspection system has detected an anomaly in your product."
    return prediction_sentence


submit = st.button(label="Submit a Leather Product Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader":
        img_file_path = uploaded_file_img
    elif input_method == "Camera Input":
        img_file_path = camera_file_img
    prediction = Anomaly_Detection(img_file_path, data_folder)
    with st.spinner(text="This may take a moment..."):
        st.write(prediction)
