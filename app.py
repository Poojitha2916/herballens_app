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
st.write("Upload a clear leaf image to identify the herbal plant, its description, and medicinal uses.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("herbal_model.keras")

model = load_model()

# ---------------- LOAD CLASS NAMES ----------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# index → plant name
class_names = {int(v): k for k, v in class_indices.items()}

# ---------------- PLANT DESCRIPTIONS ----------------
plant_descriptions = {
    "Adas": "Adas is an aromatic herb commonly used in traditional medicine. It supports digestion and relieves bloating.",
    "Aloevera": "A succulent medicinal plant known for its soothing gel. Widely used for skin care and digestive health.",
    "Amla": "A nutrient-rich fruit high in vitamin C. Commonly used to boost immunity and improve hair health.",
    "Amruta_Balli": "A climbing shrub used in Ayurveda. Known for immunity boosting and fever management properties.",
    "Ashwagandha": "A powerful adaptogenic herb used to reduce stress and improve strength and vitality.",
    "Neem": "A powerful medicinal tree known for antibacterial, antifungal, and blood-purifying properties.",
    "Tulasi": "A sacred medicinal plant in India, widely used for respiratory health and immunity.",
    "Mint": "A cooling aromatic herb commonly used to aid digestion and relieve nausea.",
    "Kunyit": "Also known as turmeric, famous for its strong anti-inflammatory and healing properties.",
    "Pepper": "A spice plant with medicinal value that improves digestion and respiratory health.",
    "Lemon_grass": "A fragrant medicinal grass used for digestion, detoxification, and stress relief.",
    "Hibiscus": "A flowering medicinal plant known for promoting hair growth and heart health.",
    "Brahmi": "A well-known brain tonic herb that enhances memory and cognitive function.",
    "Rose": "A fragrant flowering plant used in skin care, digestion, and aromatherapy.",
    "Zigzag": "An ornamental medicinal plant used in traditional herbal remedies."
}

# ---------------- PLANT USES (EXPANDED) ----------------
plant_uses = {
    "Adas": [
        "Improves digestion",
        "Relieves bloating and gas",
        "Enhances appetite",
        "Reduces stomach cramps",
        "Supports gut health"
    ],
    "Aloevera": [
        "Soothes skin irritation",
        "Heals burns and wounds",
        "Improves digestion",
        "Boosts skin hydration",
        "Supports immune system"
    ],
    "Amla": [
        "Boosts immunity",
        "Improves hair growth",
        "Enhances eyesight",
        "Supports liver health",
        "Rich source of Vitamin C"
    ],
    "Amruta_Balli": [
        "Boosts immunity",
        "Helps manage fever",
        "Detoxifies the body",
        "Improves metabolism",
        "Supports respiratory health"
    ],
    "Ashwagandha": [
        "Reduces stress and anxiety",
        "Improves strength and stamina",
        "Enhances sleep quality",
        "Boosts immunity",
        "Improves focus and memory"
    ],
    "Neem": [
        "Purifies blood",
        "Treats skin disorders",
        "Antibacterial and antifungal",
        "Improves dental health",
        "Boosts immunity"
    ],
    "Tulasi": [
        "Relieves cold and cough",
        "Boosts immunity",
        "Improves respiratory health",
        "Reduces stress",
        "Acts as antioxidant"
    ],
    "Mint": [
        "Improves digestion",
        "Relieves nausea",
        "Freshens breath",
        "Reduces headache",
        "Cooling effect on body"
    ],
    "Kunyit": [
        "Strong anti-inflammatory",
        "Promotes wound healing",
        "Boosts immunity",
        "Improves skin health",
        "Supports joint health"
    ],
    "Pepper": [
        "Improves digestion",
        "Relieves cold and cough",
        "Boosts metabolism",
        "Enhances nutrient absorption",
        "Improves respiratory health"
    ],
    "Lemon_grass": [
        "Reduces stress",
        "Improves digestion",
        "Detoxifies the body",
        "Relieves headache",
        "Improves sleep quality"
    ],
    "Hibiscus": [
        "Promotes hair growth",
        "Controls blood pressure",
        "Improves heart health",
        "Supports liver health",
        "Rich in antioxidants"
    ],
    "Brahmi": [
        "Enhances memory",
        "Improves concentration",
        "Reduces anxiety",
        "Supports brain health",
        "Improves sleep quality"
    ],
    "Rose": [
        "Improves skin hydration",
        "Reduces stress",
        "Enhances mood",
        "Supports digestion",
        "Used in aromatherapy"
    ],
    "Zigzag": [
        "Traditional medicinal plant",
        "Used in herbal remedies",
        "Supports general wellness",
        "Ornamental medicinal value",
        "Used in folk medicine"
    ]
}

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
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
            preds = model.predict(img)
            predicted_index = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0])) * 100
            plant_name = class_names[predicted_index]

        if confidence < 50:
            st.warning("⚠️ Low confidence. Please upload a clearer leaf image.")
        else:
            st.success(f"🌿 **Plant Name:** {plant_name}")
            st.info(f"🔍 **Confidence:** {confidence:.2f}%")

            # Description
            st.subheader("📖 Description")
            st.write(
                plant_descriptions.get(
                    plant_name,
                    "Description not available for this plant."
                )
            )

            # Uses
            st.subheader("🌱 Uses & Benefits")
            uses = plant_uses.get(plant_name, ["Information not available"])
            for u in uses:
                st.markdown(f"✅ {u}")
