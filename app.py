# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os
import requests
import tempfile
import cv2

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/best.pt"  # ganti sesuai nama model kamu
MODEL_URL = st.secrets.get("MODEL_URL", None)

# ---------- HELPERS ----------
@st.cache_resource
def load_model(path=MODEL_FILENAME):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if MODEL_URL:
            st.info("Model tidak ditemukan, mengunduh dari MODEL_URL...")
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model diunduh.")
        else:
            st.error("Model tidak ditemukan dan MODEL_URL tidak di-set.")
            raise FileNotFoundError("Model tidak ditemukan")

    return YOLO(path)

def read_imagefile(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)

def draw_boxes(orig_img, results, conf_thresh=0.25):
    img = orig_img.copy()
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            if conf < conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{r.names[cls]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img

# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Gunakan upload gambar atau kamera untuk mendeteksi kesegaran ikan.")

# Load model
try:
    model = load_model()
except:
    st.stop()

# Tabs: Upload atau Kamera
tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Kamera"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Upload gambar ikan (jpg, png)", type=["jpg", "jpeg", "png"])
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, step=0.05)
        imgsz = st.slider("Image size (pixels, inference)", 256, 1280, 640, step=64)

    with col2:
        st.sidebar.header("Pengaturan")
        st.sidebar.write("Model:", MODEL_FILENAME)
        if MODEL_URL:
            st.sidebar.write("Model URL: set")

    # ----- Upload Processing -----
    if uploaded:
        image_np = read_imagefile(uploaded)
        st.image(image_np, caption="Input image", use_column_width=True)

        with st.spinner("Menjalankan inferensi..."):
            results = model.predict(
                source=image_np,
                imgsz=imgsz,
                conf=conf_threshold,
                verbose=False
            )

        drawn = draw_boxes(image_np, results, conf_thresh=conf_threshold)
        st.image(drawn, caption="Deteksi", use_column_width=True)

        # Summary
        st.subheader("Ringkasan deteksi")
        detections = {}
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            names_map = r.names
            for i, conf in enumerate(boxes.conf.cpu().numpy()):
                if conf < conf_threshold:
                    continue
                cls = int(boxes.cls.cpu().numpy()[i])
                name = names_map.get(cls, str(cls))
                detections[name] = detections.get(name, 0) + 1

        if detections:
            for k, v in detections.items():
                st.write(f"- **{k}** : {v} kali")
        else:
            st.write("Tidak ada deteksi di atas threshold.")

with tab2:
    st.write("Ambil gambar dari kamera lalu deteksi otomatis.")
    camera_image = st.camera_input("📷 Ambil gambar")

    if camera_image:
        img = Image.open(camera_image).convert("RGB")
        img_np = np.array(img)
        st.image(img_np, caption="Gambar kamera", use_column_width=True)

        with st.spinner("Mendeteksi..."):
            results = model.predict(
                source=img_np,
                imgsz=640,
                conf=0.25,
                verbose=False
            )

        drawn = draw_boxes(img_np, results)
        st.image(drawn, caption="Hasil deteksi kamera", use_column_width=True)

        # Summary
        st.subheader("Ringkasan deteksi")
        detections = {}
        r = results[0]
        boxes = r.boxes
        if boxes is not None:
            names_map = r.names
            for i, conf in enumerate(boxes.conf.cpu().numpy()):
                cls = int(boxes.cls.cpu().numpy()[i])
                if conf >= 0.25:
                    name = names_map.get(cls, str(cls))
                    detections[name] = detections.get(name, 0) + 1

        if detections:
            for k, v in detections.items():
                st.write(f"- **{k}** : {v} kali")
        else:
            st.write("Tidak ada deteksi.")
