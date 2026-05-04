import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Face Mask App", layout="centered")

st.title("😷 Face Mask Detection App")
st.write("Upload image to detect With Mask / Without Mask (demo logic)")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_mask(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    results = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # 🔥 SIMPLE DEMO LOGIC (replace later with real ML model)
        avg_intensity = np.mean(face_roi)

        if avg_intensity < 100:
            label = "😷 With Mask"
            color = (0, 255, 0)
        else:
            label = "❌ Without Mask"
            color = (255, 0, 0)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append(label)

    return img, results

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    result_img, results = detect_mask(image)

    st.image(result_img, caption="Prediction Output", use_column_width=True)

    if len(results) > 0:
        for r in results:
            st.success(r)
    else:
        st.error("No face detected")