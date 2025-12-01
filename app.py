import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/best.pt"
MODEL_URL = st.secrets.get("MODEL_URL", None)

@st.cache_resource
def load_model():
    if MODEL_URL:
        from pathlib import Path
        path = Path(MODEL_FILENAME)

        if not path.exists():
            import requests
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(requests.get(MODEL_URL).content)

    return YOLO(MODEL_FILENAME)

model = load_model()


# ---------- DETEKSI GAMBAR ----------
def detect_image(image):
    results = model.predict(image, conf=0.25)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # Perbaikan warna
    return annotated


# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Upload satu atau beberapa foto ikan untuk mendeteksi kesegaran.")

uploaded_files = st.file_uploader(
    "Upload gambar ikan",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------- PROSES ----------
if uploaded_files:
    st.write(f"📸 Jumlah gambar ter-upload: **{len(uploaded_files)}**")

    for file in uploaded_files:
        st.subheader(f"Gambar: {file.name}")

        img = Image.open(file).convert("RGB")

        st.image(img, caption="Gambar asli", use_column_width=True)

        with st.spinner("Mendeteksi..."):
            result_img = detect_image(img)

        st.image(result_img, caption="Hasil deteksi", use_column_width=True)
        st.markdown("---")
