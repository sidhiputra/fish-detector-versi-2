import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ============================================
# CONFIG
# ============================================
MODEL_FILENAME = "weights/best.pt"
MODEL_URL = st.secrets.get("MODEL_URL", None)

st.set_page_config(page_title="Fish Freshness Detector", layout="wide")


# ============================================
# LOAD MODEL
# ============================================
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


# ============================================
# DETEKSI GAMBAR
# ============================================
def detect_image(image):
    results = model.predict(image, conf=0.25)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)  # Fix warna
    return annotated


# ============================================
# UI HEADER
# ============================================
st.markdown(
    """
    <h1 style='text-align:center; color:#0A6C9F;'>
        🐟 Fish Freshness Detector (YOLOv8m)
    </h1>

    <p style='text-align:center; font-size:18px;'>
        Sistem deteksi otomatis untuk menentukan kesegaran ikan menggunakan model YOLOv8m.
    </p>

    <hr style="border: 1px solid #ccc;">
    """,
    unsafe_allow_html=True
)


# ============================================
# PROSEDUR PENGGUNAAN
# ============================================
with st.expander("📘 **Prosedur Penggunaan Aplikasi**", expanded=True):
    st.markdown(
        """
        **1. Siapkan foto ikan**  
        - Ambil gambar ikan dari jarak yang cukup dekat  
        - Pastikan area ikan terlihat jelas dan tidak blur  

        **2. Upload gambar**  
        - Tekan tombol **Upload gambar**  
        - Anda bisa memilih **banyak gambar sekaligus**  

        **3. Sistem akan otomatis melakukan deteksi**  
        - Tidak perlu menekan tombol apa pun  
        - Bounding box dan label akan muncul di setiap gambar  

        **4. Baca hasil deteksi**  
        - Periksa label seperti `Fresh-Eye`, `Fresh-Skin`, `NonFresh-Eye`, `NonFresh-Skin`, `VeryFresh-Eye`, atau `VeryFresh-Skin`  
        - Semakin tinggi confidence → semakin kuat prediksinya  
        """
    )


# ============================================
# UPLOAD FILE
# ============================================
st.markdown("### 📤 Upload Foto Ikan")
uploaded_files = st.file_uploader(
    "Pilih satu atau beberapa gambar ikan",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================
# HASIL DETEKSI
# ============================================
if uploaded_files:
    st.success(f"📸 {len(uploaded_files)} gambar berhasil di-upload.")

    # Grid responsive (2 kolom)
    cols = st.columns(2)

    for idx, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")
        col = cols[idx % 2]

        with col:
            st.markdown(
                f"""
                <div style="padding:10px; background:#F7F9FB; border-radius:10px;">
                    <h4 style="color:#0A6C9F;">📎 {file.name}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(img, caption="Gambar asli", use_column_width=True)

            with st.spinner("Mendeteksi..."):
                detected = detect_image(img)

            st.image(
                detected,
                caption="Hasil Deteksi",
                use_column_width=True
            )

            st.markdown("---")

else:
    st.info("Silakan upload gambar untuk mulai deteksi.")
