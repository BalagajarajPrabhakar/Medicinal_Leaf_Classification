import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 128
MODEL_PATH = "leaf_cnn_model.h5"
CLASS_NAMES_PATH = "class_names.npy"

# -----------------------------
# LOAD MODEL (Load Once)
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)
    return model, class_names

model, class_names = load_model()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def preprocess_image(image):
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


def predict_image(image):
    img = preprocess_image(image)
    pred_prob = model.predict(img, verbose=0)[0]
    pred_class = np.argmax(pred_prob)
    confidence = pred_prob[pred_class]
    return class_names[pred_class], confidence


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("Medicinal Leaf Classification CNN")
tabs = st.tabs(["Upload Image", "Camera Capture"])

# -----------------------------
# TAB 1: UPLOAD IMAGE
# -----------------------------
with tabs[0]:
    st.header("Upload Leaf Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict", key="upload_predict"):
            pred, conf = predict_image(image)
            st.success(f"Prediction: {pred}")
            st.info(f"Confidence: {conf:.2f}")


# -----------------------------
# TAB 2: CAMERA CAPTURE (Cloud + Mobile Supported)
# -----------------------------
with tabs[1]:
    st.header("Webcam Leaf Capture")

    st.info("On mobile devices, use the camera switch option to change front/back camera.")

    # Streamlit native camera input
    camera_image = st.camera_input("Capture Leaf Image")

    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)

        if st.button("Capture & Predict", key="camera_predict"):
            pred, conf = predict_image(image)
            st.success(f"Prediction: {pred}")
            st.info(f"Confidence: {conf:.2f}")