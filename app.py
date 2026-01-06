import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HerbalLens 🌿",
    layout="centered"
)

st.title("🌿 HerbalLens – Herbal Plant Identification")
st.write("Upload a clear leaf image to identify the herbal plant")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("herbal_model.keras")

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# reverse dictionary (index → class name)
class_names = {int(v): k for k, v in class_indices.items()}

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)   # ✅ MUST match training
    return image

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict 🌱"):
        with st.spinner("Identifying plant..."):

            img = preprocess_image(image)
            predictions = model.predict(img)

            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0])) * 100
            plant_name = class_names[predicted_class]

        # ---------------- OUTPUT ----------------
        if confidence < 50:
            st.warning("⚠️ Low confidence prediction. Please upload a clearer leaf image.")
        else:
            st.success(f"🌿 **Plant Name:** {plant_name}")
            st.info(f"🔍 **Confidence:** {confidence:.2f}%")

        # ---------------- DEBUG (TEMPORARY) ----------------
        with st.expander("🔍 Debug Info"):
            st.write("Model input shape:", model.input_shape)
            st.write("Model output shape:", model.output_shape)
            st.write("Prediction vector:", predictions[0])
            st.write("Predicted index:", predicted_class)
