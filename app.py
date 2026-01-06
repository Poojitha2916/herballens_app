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
st.write("Upload a clear leaf image to identify the herbal plant and its uses")

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

# ---------------- PLANT USES DICTIONARY ----------------
plant_uses = {
    "Adas": ["Digestive aid", "Relieves bloating", "Improves appetite"],
    "Aloevera": ["Skin care", "Burn healing", "Digestive health"],
    "Amla": ["Vitamin C rich", "Boosts immunity", "Hair health"],
    "Amruta_Balli": ["Immunity booster", "Fever management", "Detoxification"],
    "Andong Merah": ["Anti-inflammatory", "Wound healing"],
    "Arali": ["Used in traditional remedies", "Supports wellness"],
    "Ashoka": ["Gynecological health", "Menstrual regulation"],
    "Ashwagandha": ["Stress relief", "Strength & vitality"],
    "Avacado": ["Heart health", "Rich in healthy fats"],
    "Bamboo": ["Anti-inflammatory", "Bone health"],
    "Basale": ["Improves digestion", "Anti-inflammatory"],
    "Belimbing Wulu": ["Antioxidant", "Digestive aid"],
    "Beluntas": ["Body odor control", "Digestive health"],
    "Betadin": ["Antiseptic plant", "Wound care"],
    "Betel": ["Oral health", "Digestive stimulant"],
    "Betel_Nut": ["Digestive stimulant", "Traditional chewing plant"],
    "Brahmi": ["Memory enhancement", "Brain tonic"],
    "Castor": ["Laxative", "Joint pain relief"],
    "Cincau Perdu": ["Cooling agent", "Digestive relief"],
    "Curry_Leaf": ["Controls diabetes", "Improves digestion"],
    "Daun Afrika": ["Anti-cancer properties", "Immunity support"],
    "Daun Cabe Jawa": ["Digestive aid", "Anti-inflammatory"],
    "Daun Cocor Bebek": ["Wound healing", "Anti-inflammatory"],
    "Daun Kumis Kucing": ["Kidney health", "Diuretic"],
    "Daun Mangkokan": ["Hair growth", "Skin care"],
    "Daun Suji": ["Natural coloring", "Digestive aid"],
    "Daun Ungu": ["Hemorrhoid relief", "Anti-inflammatory"],
    "Dewa Ndaru": ["Traditional healing plant", "General wellness"],
    "Doddapatre": ["Cold relief", "Digestive aid"],
    "Ekka": ["Pain relief", "Anti-inflammatory"],
    "Gandarusa": ["Rheumatic pain relief", "Anti-inflammatory"],
    "Ganike": ["Traditional herbal remedy", "General wellness"],
    "Garut": ["Digestive health", "Energy booster"],
    "Gauva": ["Controls diarrhea", "Rich in antioxidants"],
    "Geranium": ["Aromatherapy", "Skin care"],
    "Henna": ["Cooling effect", "Hair & skin health"],
    "Hibiscus": ["Hair growth", "Blood pressure regulation"],
    "Honge": ["Skin diseases", "Anti-inflammatory"],
    "Honje": ["Digestive aid", "Anti-oxidant"],
    "Iler": ["Anti-inflammatory", "Wound healing"],
    "Insulin": ["Blood sugar control", "Diabetes management"],
    "Jahe": ["Digestive aid", "Reduces nausea"],
    "Jasmine": ["Stress relief", "Aromatherapy"],
    "Jeruk Nipis": ["Vitamin C source", "Detoxification"],
    "Kapulaga": ["Improves digestion", "Freshens breath"],
    "Kayu Putih": ["Cold relief", "Muscle pain relief"],
    "Kecibling": ["Urinary health", "Diuretic"],
    "Kemangi": ["Digestive aid", "Anti-bacterial"],
    "Kembang Sepatu": ["Hair care", "Blood pressure control"],
    "Kenanga": ["Stress relief", "Aromatherapy"],
    "Kunyit": ["Anti-inflammatory", "Wound healing"],
    "Lampes": ["Traditional herbal medicine", "General wellness"],
    "Legundi": ["Respiratory health", "Anti-inflammatory"],
    "Lemon": ["Vitamin C", "Detoxification"],
    "Lemon_grass": ["Stress reduction", "Digestive aid"],
    "Lidah Buaya": ["Skin care", "Digestive health"],
    "Mahkota Dewa": ["Anti-cancer properties", "Blood purification"],
    "Mango": ["Digestive enzymes", "Immunity booster"],
    "Melati": ["Stress relief", "Skin care"],
    "Meniran": ["Liver protection", "Immunity booster"],
    "Mint": ["Digestive aid", "Cold relief"],
    "Murbey": ["Blood sugar regulation", "Antioxidant"],
    "Nagadali": ["Traditional remedy", "General wellness"],
    "Neem": ["Blood purification", "Skin diseases"],
    "Nilam": ["Aromatherapy", "Skin care"],
    "Nithyapushpa": ["Traditional medicine", "General wellness"],
    "Nooni": ["Digestive aid", "Immunity booster"],
    "Pacing Petul": ["Anti-inflammatory", "Pain relief"],
    "Pandan": ["Digestive aid", "Aromatherapy"],
    "Pappaya": ["Digestive enzymes", "Gut health"],
    "Patah Tulang": ["Bone healing", "Anti-inflammatory"],
    "Pecut Kuda": ["Anti-inflammatory", "Traditional medicine"],
    "Pepper": ["Improves digestion", "Cold relief"],
    "Pomegranate": ["Antioxidant rich", "Heart health"],
    "Raktachandini": ["Skin diseases", "Blood purifier"],
    "Rose": ["Skin hydration", "Aromatherapy"],
    "Saga Manis": ["Respiratory health", "Cough relief"],
    "Sapota": ["Energy booster", "Digestive health"],
    "Secang": ["Blood purification", "Anti-oxidant"],
    "Sereh": ["Digestive aid", "Stress relief"],
    "Sirih": ["Oral health", "Anti-bacterial"],
    "Srikaya": ["Digestive aid", "Antioxidant"],
    "Tin": ["Digestive health", "Rich in fiber"],
    "Tulasi": ["Cold & cough", "Immunity booster"],
    "Wood_sorel": ["Cooling agent", "Digestive aid"],
    "Zigzag": ["Traditional ornamental medicinal plant"]
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
            predictions = model.predict(img)

            predicted_class = int(np.argmax(predictions[0]))
            confidence = float(np.max(predictions[0])) * 100
            plant_name = class_names[predicted_class]

        if confidence < 50:
            st.warning("⚠️ Low confidence prediction. Please upload a clearer leaf image.")
        else:
            st.success(f"🌿 **Plant Name:** {plant_name}")
            st.info(f"🔍 **Confidence:** {confidence:.2f}%")

            st.subheader("🌱 Uses & Benefits")
            uses = plant_uses.get(plant_name, ["Information not available"])
            for use in uses:
                st.write(f"• {use}")
