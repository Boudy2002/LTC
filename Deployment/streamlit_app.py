import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Load model and class labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Deployment/model.h5', compile=False)
    return model

@st.cache_data
def load_class_labels():
    with open('Deployment/class_labels.json', 'r') as f:
        return json.load(f)

model = load_model()
class_labels = load_class_labels()

# Set page config
st.set_page_config(page_title="Land Type Classifier", layout="centered")

st.title("🌍 Land Type Classification using Satellite Images")
st.write("Upload a satellite image and we'll predict the land category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {class_labels[str(predicted_index)]}")
    st.info(f"**Confidence:** {confidence:.2%}")