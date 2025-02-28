import pickle
import streamlit as st
import numpy as np
import base64

# -------------------- Function to Set Background Image --------------------
def set_bg(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:images/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main-title {{
            text-align: center;
            color: white;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 2px 2px 10px black;
        }}
        .sub-title {{
            text-align: center;
            color: blue;
            font-size: 20px;
            text-shadow: 1px 1px 5px black;
        }}
        .stNumberInput>div>div>input {{
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
            border: 2px solid #4CAF50;
        }}
        .stButton>button {{
            background-color: #28a745;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #218838;
            text_align:center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("images.jpg")

# -------------------- Load the Trained Model --------------------
@st.cache_resource
def load_model():
    try:
        with open("nb_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -------------------- Streamlit UI --------------------
st.markdown("<h1 class='main-title'>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter soil details to get the best crop suggestion.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- User Input Fields --------------------
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌾 Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("🌿 Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("🌾 Potassium (K)", min_value=0, max_value=200, value=50)
    ph = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=7.0)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=200.0, value=50.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=900.0, value=100.0)

st.markdown("---")

# -------------------- Prediction Button --------------------
if st.button("🌾 Recommend"):
    if model is not None:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        with st.spinner("🔄 Predicting... Please wait"):
            try:
                prediction = model.predict(features)[0]

                # Stylish Prediction Output
                st.markdown(
                    f"""
                    <div style="
                        border: 4px solid #4CAF50;
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        background-color: #f1f8e9;
                        color: #1B5E20;
                        font-size: 28px;
                        font-weight: bold;
                        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
                        margin: 20px auto;
                        width: 90%;
                    ">
                        🌱 Recommended Crop is: <span style="color: #2E7D32;">{prediction}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
    else:
        st.error("🚨 Model not loaded. Please check `nb_model.pkl`.")
