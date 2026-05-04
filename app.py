import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😷 Face Mask Detection App (Simple Version)")
st.write("Upload an image to detect face (demo version without TensorFlow)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_face(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img, len(faces)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    result_img, face_count = detect_face(image)

    st.image(result_img, caption="Detected Faces", use_column_width=True)

    if face_count > 0:
        st.success(f"Faces detected: {face_count}")
    else:
        st.error("No face detected")