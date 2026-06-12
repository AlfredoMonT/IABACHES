import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog as skimage_hog

st.set_page_config(page_title="Deteccion de Baches - UAC", layout="wide")

st.title("Sistema de Vision Computacional para Deteccion de Baches")
st.caption("Universidad Andina del Cusco - Ingenieria de Sistemas 2026")
st.divider()

FASE1_MODEL = "models/saved/random_forest_fase1.pkl"
YOLO_MODEL = "models/saved/best_yolo11n.pt"
RTDETR_MODEL = "models/saved/best_rtdetr.pt"

@st.cache_resource
def load_fase1():
    import joblib
    if os.path.exists(FASE1_MODEL):
        return joblib.load(FASE1_MODEL)
    return None

@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    if os.path.exists(YOLO_MODEL):
        return YOLO(YOLO_MODEL)
    return None

@st.cache_resource
def load_rtdetr():
    from ultralytics import RTDETR
    if os.path.exists(RTDETR_MODEL):
        return RTDETR(RTDETR_MODEL)
    return None

st.sidebar.title("Configuracion")
modelo_sel = st.sidebar.radio(
    "Modelo:",
    [
        "Fase 1 - Random Forest + HOG",
        "Fase 2 - YOLO11n (83 FPS)",
        "Fase 3 - RT-DETR (28 FPS)",
    ],
)
confianza = st.sidebar.slider("Umbral de confianza (Fases 2 y 3)", 0.1, 0.9, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.markdown(
    "Comparativa de modelos\n\n"
    "| Modelo | F1 | mAP50 | FPS |\n"
    "|---|---|---|---|\n"
    "| Random Forest | 62.4% | - | 9 |\n"
    "| YOLO11n | 79.2% | 77.6% | 83 |\n"
    "| RT-DETR | 85.1% | 84.3% | 28 |"
)

uploaded = st.file_uploader("Sube una imagen de la calle", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    img_np = np.array(img_pil)

    if "Random Forest" in modelo_sel:
        model = load_fase1()
        if model is None:
            st.error("Modelo no encontrado. Ejecuta primero scripts/run_pipeline_fase1.py")
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            patch = cv2.resize(gray, (64, 64))
            features = skimage_hog(
                patch.astype(np.float32) / 255.0,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            ).reshape(1, -1)
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            conf = proba[pred] * 100
            label = "BACHE / REJILLA DETECTADO" if pred == 1 else "ASFALTO NORMAL"
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_pil, caption="Imagen Original", use_container_width=True)
            with col2:
                st.image(cv2.resize(gray, (img_np.shape[1], img_np.shape[0])),
                         caption="Preprocesado (escala de grises + CLAHE)",
                         use_container_width=True)
            st.divider()
            if pred == 1:
                st.error(f"{label} - Confianza: {conf:.1f}%")
            else:
                st.success(f"{label} - Confianza: {conf:.1f}%")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("P(Bache)", f"{proba[1]*100:.1f}%")
            with col_b:
                st.metric("P(Normal)", f"{proba[0]*100:.1f}%")

    else:
        if "YOLO" in modelo_sel:
            model = load_yolo()
            model_name = "YOLO11n"
        else:
            model = load_rtdetr()
            model_name = "RT-DETR"

        if model is None:
            st.error(f"Modelo no encontrado. Ejecuta primero scripts/run_pipeline_fase2y3.py")
        else:
            with st.spinner(f"Analizando con {model_name}..."):
                results = model(img_np, conf=confianza)[0]
                annotated = results.plot()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagen Original")
                st.image(img_pil, use_container_width=True)
            with col2:
                st.subheader(f"Deteccion - {model_name}")
                st.image(annotated[:, :, ::-1], use_container_width=True)

            boxes = results.boxes
            st.divider()
            if boxes is not None and len(boxes) > 0:
                st.error(f"Se detectaron {len(boxes)} anomalias en la imagen")
                cols = st.columns(min(len(boxes), 4))
                for i, box in enumerate(boxes):
                    c = float(box.conf[0]) * 100
                    with cols[i % 4]:
                        st.metric(f"Anomalia {i+1}", f"{c:.1f}%", "confianza")
            else:
                st.success("No se detectaron baches ni rejillas en esta imagen")
else:
    st.info("Sube una imagen para comenzar el analisis")
