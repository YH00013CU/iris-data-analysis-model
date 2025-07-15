import streamlit as st
import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class_names = ["Setosa", "Versicolor", "Virginica"]
class_emojis = ["🌼", "🌺", "🌸"]

# Verified direct image links (resized via Wikimedia)
flower_images = {
    "Setosa": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Iris_setosa_2.jpg/640px-Iris_setosa_2.jpg",
    "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/640px-Iris_versicolor_3.jpg",
    "Virginica": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/640px-Iris_virginica.jpg"
}

# Streamlit config
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌷")

# Styling for dark text
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #fdf6f9, #e8f5e9);
        color: #111111;
    }
    html, body, [class*="css"] {
        color: #111111;
    }
    .stButton>button {
        background-color: #ec407a;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# UI layout
st.title("🌷 Iris Flower Predictor")
st.subheader("Let’s predict your flower species 🌸")
st.write("Adjust the sliders below to match your Iris flower’s measurements and let the magic bloom.")

# Feature sliders
sepal_len = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_wid = st.slider("🍃 Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_len = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 1.4)
petal_wid = st.slider("🌼 Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction button
if st.button("💐 Predict Flower"):
    inputs = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    prediction = model.predict(inputs)[0]
    name = class_names[prediction]
    emoji = class_emojis[prediction]

    st.success(f"Your flower is: **{name}** {emoji}")
    st.image(flower_images[name], caption=f"{name} Flower", use_container_width=True)
