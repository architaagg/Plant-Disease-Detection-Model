import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # REQUIRED: missing import

st.set_page_config(page_title="Plant Disease Detector 🌱", layout="wide")

# Load model once (do this outside the functions to avoid reloading)
@st.cache_resource

def load_model():
    model = tf.keras.models.load_model(r'C:\Drive E_F\NIT Jalandhar\2nd Year\4th Semester\IML Lab\Extraplant_disease_prediction_model.h5')  # Change path if needed
    return model

model = load_model()

# TensorFlow Model Prediction
def load_and_preprocess_image(image_data, target_size=(224, 224)):
    img = Image.open(image_data).convert('RGB')  # Ensures consistent 3 channels
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


def predict_image_class(image_data):
    preprocessed_img = load_and_preprocess_image(image_data)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index

# Class names (hardcoded, should match training order)
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""Welcome to the Plant Disease Recognition System! Upload a plant leaf image to identify diseases.""")

# About
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This app uses a deep learning model trained on the PlantVillage dataset.
    The dataset includes 38 classes of healthy and diseased leaves across various crops.
    """)

# Prediction
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_index = predict_image_class(test_image)
                st.success(f"🩺 Model predicts: **{class_name[result_index]}**")
