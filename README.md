# 🛣️ Sistema de Visión Computacional para la Detección Automatizada de Baches y Rejillas

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Universidad Andina del Cusco](https://img.shields.io/badge/UAC-Ingeniería%20de%20Sistemas-red.svg)](https://uandina.edu.pe/)

## 📌 Descripción

Proyecto integrador del curso **Inteligencia Artificial Aplicada** de la Escuela Profesional de Ingeniería de Sistemas de la **Universidad Andina del Cusco (UAC)**. El sistema desarrolla un pipeline de visión por computadora para detectar y clasificar automáticamente baches y anomalías viales en imágenes de infraestructura urbana, apoyando el mantenimiento preventivo de vías públicas en Cusco, Perú.

El proyecto se desarrolla en **3 fases progresivas**:
- **Fase 1:** Clasificación con Random Forest + extracción de características HOG
- **Fase 2:** Detección con Redes Neuronales Convolucionales (CNN)
- **Fase 3:** Detección en tiempo real con YOLOv11 + Transformers

---

## 👥 Equipo

| Nombre | Rol |
|---|---|
| Orellana Cusihuaman Luis Anthony | Ingeniero de Datos y Preprocesamiento de Imágenes |
| Montufar Diaz Alfredo Gerardo | Especialista en Modelado de Machine Learning |
| Vilca Ramos Luis Gerardo | Analista de Métricas de Visión y Despliegue |

---

## 🗂️ Estructura del Repositorio

```
IABACHES/
│
├── data/                        # Datos del proyecto (no se suben imágenes al repo)
│   ├── raw/                     # Imágenes originales descargadas de Kaggle
│   └── processed/               # Imágenes preprocesadas y divididas
│       ├── train/               # 70% del dataset para entrenamiento
│       │   ├── pothole/         # Imágenes con bache
│       │   └── normal/          # Imágenes de asfalto normal
│       ├── val/                 # 15% para validación durante entrenamiento
│       │   ├── pothole/
│       │   └── normal/
│       └── test/                # 15% para evaluación final del modelo
│           ├── pothole/
│           └── normal/
│
├── notebooks/                   # Jupyter Notebooks por fase (exploración y experimentación)
│   ├── fase1_exploracion_datos.ipynb
│   ├── fase1_extraccion_hog.ipynb
│   ├── fase1_entrenamiento_rf.ipynb
│   ├── fase2_cnn_entrenamiento.ipynb    # (Fase 2 - próximamente)
│   └── fase3_yolo_deteccion.ipynb       # (Fase 3 - próximamente)
│
├── src/                         # Código fuente modular y reutilizable
│   ├── preprocessing/           # Scripts de limpieza y preparación de imágenes
│   │   ├── __init__.py
│   │   └── image_processor.py
│   ├── features/                # Extracción de características (HOG, GLCM, etc.)
│   │   ├── __init__.py
│   │   └── hog_extractor.py
│   ├── models/                  # Definición, entrenamiento y guardado de modelos
│   │   ├── __init__.py
│   │   └── random_forest_model.py
│   ├── evaluation/              # Métricas, matrices de confusión y reportes
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/                   # Funciones auxiliares compartidas
│       ├── __init__.py
│       └── helpers.py
│
├── models/                      # Modelos entrenados guardados (.pkl, .h5, .pt)
│   └── saved/
│       └── .gitkeep
│
├── reports/                     # Resultados, gráficas y reportes generados
│   ├── figures/                 # Imágenes de curvas de aprendizaje, matrices, etc.
│   └── phase1/                  # Informe y entregables de la Fase 1
│
├── scripts/                     # Scripts ejecutables de línea de comandos
│   ├── download_dataset.py      # Descarga el dataset desde Kaggle API
│   └── run_pipeline.py          # Ejecuta el pipeline completo de la Fase 1
│
├── docs/                        # Documentación técnica adicional
│   └── architecture_diagram.md
│
├── requirements.txt             # Dependencias del proyecto
├── .gitignore                   # Archivos y carpetas ignorados por Git
└── README.md                    # Este archivo
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/AlfredoMonT/IABACHES.git
cd IABACHES
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Descargar el dataset
```bash
# Configura tu API key de Kaggle primero (ver docs/kaggle_setup.md)
python scripts/download_dataset.py
```

### 4. Ejecutar el pipeline de Fase 1
```bash
python scripts/run_pipeline.py
```

### 5. O explorar con los notebooks
```bash
jupyter notebook notebooks/fase1_exploracion_datos.ipynb
```

---

## 📊 Dataset

- **Fuente:** [Pothole Detection Dataset: YOLOv11 Optimized](https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized) — Kaggle
- **Total de imágenes:** 3,940
- **Clases:** `pothole` (bache/rejilla) | `normal` (asfalto sin anomalía)
- **División:** 70% train / 15% val / 15% test

> ⚠️ Las imágenes **no se incluyen** en este repositorio por su tamaño. Descárgalas con el script `scripts/download_dataset.py`.

---

## 🧠 Fases del Proyecto

### ✅ Fase 1 — Clasificación con Random Forest (actual)
- Preprocesamiento con OpenCV (resize, grayscale, CLAHE)
- Extracción de características con HOG (Histogram of Oriented Gradients)
- Clasificación binaria con Random Forest (Scikit-Learn)
- Métricas: F1-Score, Accuracy, AUC-ROC, Matriz de Confusión

### 🔄 Fase 2 — CNN con Deep Learning (próximo)
- Entrenamiento con TensorFlow/Keras
- Transfer Learning con ResNet50 / VGG16
- Comparación de métricas con Fase 1

### 🔄 Fase 3 — Detección en Tiempo Real (próximo)
- YOLOv11 para detección de objetos con bounding boxes
- Vision Transformer (ViT) para análisis semántico
- Demo en video con detección en tiempo real

---

## 📈 Resultados Fase 1

> _Resultados pendientes de experimentación. Se actualizarán al completar el entrenamiento._

| Modelo | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Random Forest + HOG | TBD | TBD | TBD |

---

## 🔬 Referencias

1. [Pothole Detection and Recognition based on Transfer Learning (arXiv, 2025)](https://arxiv.org/)
2. [Road Surface Condition Detection with Machine Learning using DOT Cameras (arXiv, 2025)](https://arxiv.org/)
3. [Computer Vision-Based Detection and Classification of Road Obstacles (IEEE, 2024)](https://ieeexplore.ieee.org/)
4. [Deep Learning Enhanced feature extraction of Potholes Using Vision and LiDAR (IEEE, 2024)](https://ieeexplore.ieee.org/)
5. [Enhancing Pothole Detection and Characterization: Integrated Segmentation (arXiv, 2025)](https://arxiv.org/)

---

## 📄 Licencia

MIT License — ver archivo [LICENSE](LICENSE)

---

*Universidad Andina del Cusco — Escuela Profesional de Ingeniería de Sistemas — 2026*
