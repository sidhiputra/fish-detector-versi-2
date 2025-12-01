import streamlit as st
import torch
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
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # perbaikan warna
    return annotated


# ---------- DETEKSI REALTIME ----------
def detect_realtime():
    st.info("Mengaktifkan kamera... Klik STOP untuk menghentikan.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("❌ Kamera tidak ditemukan.")
        return

    frame_window = st.image([])

    stop_button = st.button("STOP")

    while True:
        ret, frame = camera.read()
        if not ret:
            st.warning("Gagal membaca frame kamera")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model.predict(rgb, conf=0.25)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_window.image(annotated, channels="RGB")

        if stop_button:
            break

    camera.release()


# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Detect kesegaran ikan menggunakan model YOLOv8 Anda.")

mode = st.radio("Pilih Mode:", ["Upload Image", "Realtime Detection"])


# ---------- MODE UPLOAD ----------
if mode == "Upload Image":
    uploaded_files = st.file_uploader(
        "Upload satu atau beberapa gambar ikan",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"📸 Jumlah gambar: **{len(uploaded_files)}**")

        for file in uploaded_files:
            st.subheader(f"Gambar: {file.name}")

            img = Image.open(file)
            img = img.convert("RGB")

            st.image(img, caption="Gambar asli", use_column_width=True)

            with st.spinner("Mendeteksi..."):
                result_img = detect_image(img)

            st.image(result_img, caption="Hasil deteksi", use_column_width=True)
            st.markdown("---")


# ---------- MODE REALTIME ----------
elif mode == "Realtime Detection":
    st.write("Klik tombol di bawah untuk mulai deteksi realtime menggunakan webcam.")
    if st.button("Mulai Realtime"):
        detect_realtime()
