import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# 🔥 Recreate CNN model architecture (same as training)
model = Sequential([
    Input(shape=(128,128,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ✅ Load weights (IMPORTANT: filename must match)
model.load_weights("cnn_model.weights.h5")

# ⚠️ Update labels based on your dataset
labels = {0: "Without Mask", 1: "With Mask"}

# 🎯 Streamlit UI
st.title("😷 Face Mask Detection App")
st.write("Upload an image to check if a person is wearing a mask")

# 📤 Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# 🧠 Image preprocessing
def preprocess(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 🔍 Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img)[0][0]

    result = labels[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")