import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HerbalLens 🌿", layout="centered")

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

class_names = {int(v): k for k, v in class_indices.items()}

# ---------------- PLANT DESCRIPTIONS (ALL 86) ----------------
plant_descriptions = {
    "Adas": "Adas is an aromatic herb commonly used in traditional medicine. It supports digestion and relieves bloating.",
    "Aloevera": "A succulent medicinal plant known for its soothing gel. Widely used for skin care and digestive health.",
    "Amla": "A nutrient-rich fruit high in vitamin C. Commonly used to boost immunity and improve hair health.",
    "Amruta_Balli": "A climbing shrub used in Ayurveda. Known for immunity boosting and fever management properties.",
    "Andong Merah": "A traditional medicinal plant used in herbal remedies. Known for anti-inflammatory and wound healing effects.",
    "Arali": "An ornamental plant with medicinal value. Traditionally used for general wellness.",
    "Ashoka": "A sacred medicinal tree in Ayurveda. Commonly used for gynecological and menstrual health.",
    "Ashwagandha": "A powerful adaptogenic herb. Used to reduce stress and improve strength and vitality.",
    "Avacado": "A nutritious fruit rich in healthy fats. Supports heart health and overall nutrition.",
    "Bamboo": "A fast-growing plant used in herbal medicine. Known for bone strengthening and anti-inflammatory properties.",
    "Basale": "A leafy vegetable with medicinal benefits. Helps digestion and reduces inflammation.",
    "Belimbing Wulu": "A medicinal fruit plant rich in antioxidants. Commonly used for digestive health.",
    "Beluntas": "A shrub used in traditional medicine. Helps control body odor and supports digestion.",
    "Betadin": "A medicinal plant with antiseptic properties. Traditionally used for wound care.",
    "Betel": "A widely used medicinal leaf in Asia. Supports oral health and digestion.",
    "Betel_Nut": "A traditional chewing nut. Used as a digestive stimulant in herbal practices.",
    "Brahmi": "A well-known brain tonic herb. Enhances memory and cognitive function.",
    "Castor": "A medicinal plant known for castor oil. Used as a laxative and for joint pain relief.",
    "Cincau Perdu": "A cooling medicinal plant. Helps in digestion and body cooling.",
    "Curry_Leaf": "An aromatic leaf used in cooking and medicine. Helps regulate digestion and blood sugar.",
    "Daun Afrika": "A medicinal plant used in folk medicine. Known for immunity support and anti-cancer properties.",
    "Daun Cabe Jawa": "A traditional herbal plant. Used to improve digestion and reduce inflammation.",
    "Daun Cocor Bebek": "A succulent medicinal plant. Known for wound healing and anti-inflammatory effects.",
    "Daun Kumis Kucing": "A kidney-supporting medicinal plant. Acts as a natural diuretic.",
    "Daun Mangkokan": "A leafy medicinal plant. Commonly used for hair growth and skin care.",
    "Daun Suji": "A natural coloring plant with medicinal value. Supports digestion.",
    "Daun Ungu": "A traditional herbal plant. Used to treat hemorrhoids and inflammation.",
    "Dewa Ndaru": "A rare medicinal plant in folk medicine. Used for general health improvement.",
    "Doddapatre": "An aromatic medicinal herb. Commonly used to treat cold and digestion issues.",
    "Ekka": "A traditional medicinal shrub. Used for pain relief and inflammation.",
    "Gandarusa": "A medicinal plant used for joint pain. Known for anti-inflammatory effects.",
    "Ganike": "A lesser-known herbal plant. Used in traditional wellness treatments.",
    "Garut": "A tuber plant used in herbal food remedies. Helps digestion and boosts energy.",
    "Gauva": "A fruit-bearing medicinal tree. Used to control diarrhea and boost immunity.",
    "Geranium": "An aromatic medicinal plant. Used in skin care and aromatherapy.",
    "Henna": "A natural dye plant with cooling properties. Supports hair and skin health.",
    "Hibiscus": "A flowering medicinal plant. Promotes hair growth and heart health.",
    "Honge": "A medicinal tree used in Ayurveda. Known for treating skin diseases.",
    "Honje": "A traditional herbal plant. Supports digestion and antioxidant activity.",
    "Iler": "A medicinal ornamental plant. Used for wound healing and inflammation.",
    "Insulin": "A medicinal plant used for diabetes management. Helps regulate blood sugar levels.",
    "Jahe": "Commonly known as ginger. Widely used for digestion and nausea relief.",
    "Jasmine": "A fragrant flowering plant. Used for relaxation and aromatherapy.",
    "Jeruk Nipis": "A citrus medicinal fruit. Rich in vitamin C and detoxifying properties.",
    "Kapulaga": "Aromatic spice plant. Improves digestion and freshens breath.",
    "Kayu Putih": "A medicinal tree producing eucalyptus oil. Used for cold and muscle pain relief.",
    "Kecibling": "A medicinal herb supporting urinary health. Acts as a diuretic.",
    "Kemangi": "An aromatic herbal leaf. Supports digestion and fights bacteria.",
    "Kembang Sepatu": "A flowering medicinal plant. Used for hair care and blood pressure control.",
    "Kenanga": "A fragrant medicinal flower. Used in stress relief and aromatherapy.",
    "Kunyit": "Also known as turmeric. Famous for anti-inflammatory and healing properties.",
    "Lampes": "A traditional medicinal plant. Used in folk remedies for wellness.",
    "Legundi": "A herbal plant used for respiratory health. Reduces inflammation.",
    "Lemon": "A citrus medicinal fruit. Used for detoxification and immunity.",
    "Lemon_grass": "A fragrant medicinal grass. Used for digestion and stress relief.",
    "Lidah Buaya": "Another name for aloe vera. Used for skin healing and digestion.",
    "Mahkota Dewa": "A powerful medicinal plant. Known for blood purification and anti-cancer properties.",
    "Mango": "A tropical fruit tree with medicinal value. Supports digestion and immunity.",
    "Melati": "A fragrant medicinal flower. Used in skin care and relaxation.",
    "Meniran": "A traditional herbal plant. Supports liver health and immunity.",
    "Mint": "A cooling aromatic herb. Used for digestion and cold relief.",
    "Murbey": "A medicinal fruit plant. Helps regulate blood sugar and acts as antioxidant.",
    "Nagadali": "A folk medicinal plant. Used for general wellness.",
    "Neem": "A powerful medicinal tree. Known for antibacterial and blood purifying properties.",
    "Nilam": "Aromatic medicinal plant. Used in perfumes and skin care.",
    "Nithyapushpa": "A traditional medicinal flower. Used for general health.",
    "Nooni": "Also known as noni plant. Used to boost immunity and digestion.",
    "Pacing Petul": "A medicinal herb. Used for pain relief and inflammation.",
    "Pandan": "An aromatic medicinal plant. Used in digestion and relaxation.",
    "Pappaya": "Papaya plant used medicinally. Supports digestion and gut health.",
    "Patah Tulang": "A medicinal shrub. Known for bone healing properties.",
    "Pecut Kuda": "A traditional medicinal plant. Used for inflammation and pain relief.",
    "Pepper": "A spice plant with medicinal benefits. Improves digestion and treats cold.",
    "Pomegranate": "A medicinal fruit rich in antioxidants. Supports heart health.",
    "Raktachandini": "A medicinal tree. Used for skin diseases and blood purification.",
    "Rose": "A fragrant flowering plant. Used in skin care and aromatherapy.",
    "Saga Manis": "A medicinal climber plant. Used for respiratory and cough relief.",
    "Sapota": "A fruit tree with medicinal value. Supports digestion and energy.",
    "Secang": "A medicinal wood plant. Used for blood purification and antioxidants.",
    "Sereh": "Another name for lemongrass. Used for digestion and stress relief.",
    "Sirih": "A medicinal leaf plant. Used for oral health and antibacterial purposes.",
    "Srikaya": "A fruit-bearing medicinal plant. Supports digestion and antioxidants.",
    "Tin": "Also known as fig plant. Rich in fiber splitting digestive benefits.",
    "Tulasi": "A sacred medicinal plant in India. Used for immunity and respiratory health.",
    "Wood_sorel": "A medicinal leafy plant. Used as cooling agent and digestive aid.",
    "Zigzag": "An ornamental medicinal plant. Used in traditional herbal remedies."
}

# ---------------- PLANT USES (ALL 86) ----------------
plant_uses = {
    k: [
        "Improves digestion",
        "Boosts immunity",
        "Reduces inflammation",
        "Supports overall wellness",
        "Used in traditional medicine"
    ]
    for k in plant_descriptions.keys()
}

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.expand_dims(np.array(image), axis=0)
    return preprocess_input(image)

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload a leaf image", ["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Predict 🌱"):
        pred = model.predict(preprocess_image(image))
        idx = int(np.argmax(pred[0]))
        conf = float(np.max(pred[0])) * 100
        plant = class_names[idx]

        st.success(f"🌿 Plant Name: {plant}")
        st.info(f"Confidence: {conf:.2f}%")

        st.subheader("📖 Description")
        st.write(plant_descriptions.get(plant))

        st.subheader("🌱 Uses & Benefits")
        for u in plant_uses.get(plant, []):
            st.markdown(f"✅ {u}")
