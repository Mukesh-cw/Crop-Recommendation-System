import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import os
import base64

# Load dataset
file_path = 'Crop_recommendation.csv'
try:
    crop = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Error: Crop_recommendation.csv not found! Please upload the correct file.")
    st.stop()

crop_dict = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10, 
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20, 
    'jute': 21, 'coffee': 22
}

crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)
X = crop.drop('crop_num', axis=1)
y = crop['crop_num']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize and Standardize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# Reverse crop mapping
crop_reverse_dict = {v: k for k, v in crop_dict.items()}

# Streamlit App Setup
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# Custom Sidebar Navigation
st.sidebar.markdown("""
<style>
.sidebar .sidebar-content {
    font-family: 'Roboto', sans-serif;
    font-size: 20px;
    color: #333;
}
.sidebar .sidebar-content div:hover {
    background-color: #f1f1f1;
    cursor: pointer;
}
.stButton button {
    font-family: 'Arial', sans-serif;
    font-size: 20px;
    font-weight: bold;
    padding: 15px 30px;
}
input[type="number"] {
    font-size: 1.8em;
    height: 70px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Links (simulate clickable sections)
st.sidebar.title("Navigation")
navigation = st.sidebar.selectbox("Go to", ["About", "Prediction", "Previous History"])

# Initialize session history
if 'history' not in st.session_state:
    st.session_state['history'] = []

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    return crop_reverse_dict.get(prediction, "No suitable crop found")

if navigation == "About":
    st.title("🌿 About the Crop Recommendation System")
    st.write("""
        Welcome to the **Crop Recommendation System**!
        
        This intelligent tool helps farmers, gardeners, and agricultural planners choose the best crops 
        for their land based on vital soil and environmental parameters.

        ✨ **Features:**
        - Accurate crop predictions using machine learning
        - Easy-to-use interface for quick recommendations
        - Save and review your past predictions
    """)

    # Display the 8 images in a grid (2x4 layout for example)
    image_folder = "Images/"
    crop_images = [
        "rice.jpg", "papaya.jpeg", "orange.jpeg", "blackgram2.webp",
        "coffee.jpeg", "grapes2.webp", "muskmelon.jpg", "banana.jpeg"
    ]

    cols = st.columns(4)
    for i, image_file in enumerate(crop_images):
        img_path = os.path.join(image_folder, image_file)
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode()
                cols[i % 4].markdown(
                    f"""
                    <div style='min-height:150px; display:flex; align-items:center; justify-content:center;'>
                        <img src='data:image/jpeg;base64,{img_b64}' style='max-width:100%; height:auto;'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            cols[i % 4].warning(f"Image '{image_file}' not found.")

elif navigation == "Prediction":
    st.title("🌾 Crop Prediction")
    st.subheader("Enter your soil and climate details:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50, step=1, format="%d")
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50, step=1, format="%d")
    with col2:
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50, step=1, format="%d")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, format="%.1f")
    with col4:
        rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=300, value=100, step=1, format="%d")

    if st.button("🌱 Recommend Crop"):
        recommended_crop = recommend_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"🌾 Recommended Crop: {recommended_crop}")

        st.session_state['history'].append({
            'N': N, 'P': P, 'K': K, 'Temperature': temperature,
            'Humidity': humidity, 'pH': ph, 'Rainfall': rainfall,
            'Recommended Crop': recommended_crop
        })

        image_path = f"Images/{recommended_crop.lower()}.jpeg"
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_b64 = base64.b64encode(img_bytes).decode()
                st.markdown(
                    f"""
                    <div style='min-height:200px; display:flex; align-items:center; justify-content:center;'>
                        <img src='data:image/jpeg;base64,{img_b64}' style='max-width:100%; height:auto;'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning(f"Image for '{recommended_crop}' not found. Please add 'Images/{recommended_crop.lower()}.jpeg'.")

    st.markdown(f"### 📊 Model Accuracy: **{accuracy:.2f}**")

elif navigation == "Previous History":
    st.title("📜 Previous Predictions History")

    if st.session_state['history']:
        df = pd.DataFrame(st.session_state['history'])
        st.dataframe(df)
    else:
        st.info("No previous history available. Start by making some predictions!")

