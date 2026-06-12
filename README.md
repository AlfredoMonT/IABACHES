<<<<<<< HEAD
# IABACHES - Sistema de Vision Computacional para Deteccion de Baches

Universidad Andina del Cusco - Ingenieria de Sistemas 2026

Docente: Ing. Felix Vargas Quispe

Integrantes:
- Montufar Diaz Alfredo Gerardo
- Orellana Cusihuaman Luis Anthony
- Vilca Ramos Luis Gerardo

---

## Descripcion

Sistema de vision computacional para la deteccion automatizada de baches y rejillas en entornos urbanos. El proyecto esta organizado en tres fases de complejidad creciente:

- Fase 1: Extraccion de caracteristicas HOG + clasificador Random Forest
- Fase 2: Deteccion de objetos con YOLO11n (CNN)
- Fase 3: Deteccion de objetos con RT-DETR (Transformer)

---

## Estructura del Repositorio

```
IABACHES/
├── data/
│   ├── raw/                    # Dataset descargado de Kaggle (no se sube al repo)
│   └── processed/              # Imagenes preprocesadas por fase
├── notebooks/                  # Cuadernos Jupyter por fase
├── src/
│   ├── preprocessing/          # Preprocesamiento de imagenes
│   ├── features/               # Extraccion de caracteristicas HOG
│   ├── models/                 # Definicion y entrenamiento de modelos
│   ├── evaluation/             # Metricas y visualizaciones
│   └── utils/                  # Funciones auxiliares
├── models/saved/               # Pesos entrenados (.pkl, .pt)
├── reports/                    # Graficas y metricas generadas
├── app/                        # Aplicacion Streamlit
├── scripts/                    # Scripts ejecutables
└── docs/                       # Documentacion tecnica
=======
Sistema de Visión Computacional para la Detección Automatizada de Baches y Rejillas

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Universidad Andina del Cusco](https://img.shields.io/badge/UAC-Ingeniería%20de%20Sistemas-red.svg)](https://uandina.edu.pe/)

##Descripción

Proyecto integrador del curso **Inteligencia Artificial Aplicada** de la Escuela Profesional de Ingeniería de Sistemas de la **Universidad Andina del Cusco (UAC)**. El sistema desarrolla un pipeline de visión por computadora para detectar y clasificar automáticamente baches y anomalías viales en imágenes de infraestructura urbana, apoyando el mantenimiento preventivo de vías públicas en Cusco, Perú.

El proyecto se desarrolla en **3 fases progresivas**:
- **Fase 1:** Clasificación con Random Forest + extracción de características HOG
- **Fase 2:** Detección con Redes Neuronales Convolucionales (CNN)
- **Fase 3:** Detección en tiempo real con YOLOv11 + Transformers

---

##Equipo

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
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
```

---

<<<<<<< HEAD
## Instalacion

### Requisitos previos

- Python 3.10 o superior
- Anaconda o Miniconda (recomendado) o pip

### Opcion A: Conda (recomendado)

```bash
conda create -n iabaches python=3.10
conda activate iabaches
pip install -r requirements.txt
```

### Opcion B: pip con entorno virtual

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
=======
##Instalación y Uso

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
python scripts/download_dataset.py o descargarlo de la sección de dataset.
```

### 4. Ejecutar el pipeline de Fase 1
```bash
python scripts/run_pipeline.py
```

### 5. O explorar con los notebooks
```bash
jupyter notebook notebooks/fase1_exploracion_datos.ipynb
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
```

---

<<<<<<< HEAD
## Uso

### 1. Descargar el dataset

```bash
python scripts/download_dataset.py --username TU_USUARIO_KAGGLE --key TU_API_KEY
```

### 2. Ejecutar Fase 1 (Random Forest + HOG)

```bash
python scripts/run_pipeline_fase1.py
```

Genera:
- `models/saved/random_forest_fase1.pkl`
- `reports/figures/confusion_matrix_fase1.png`
- `reports/figures/metrics_fase1.png`
- `reports/phase1/metrics_fase1.json`

### 3. Ejecutar Fase 2 y 3 (YOLO11n y RT-DETR)

Entrenar solo YOLO11n:
```bash
python scripts/run_pipeline_fase2y3.py --mode yolo --epochs 50
```

Entrenar solo RT-DETR:
```bash
python scripts/run_pipeline_fase2y3.py --mode rtdetr --epochs 50
```

Entrenar ambos y comparar:
```bash
python scripts/run_pipeline_fase2y3.py --mode both --epochs 50
```

Solo comparar (si ya tienes los pesos):
```bash
python scripts/run_pipeline_fase2y3.py --mode compare
```

Genera:
- `models/saved/best_yolo11n.pt`
- `models/saved/best_rtdetr.pt`

### 4. Lanzar la aplicacion Streamlit

```bash
streamlit run app/app.py
```

Abre en el navegador: `http://localhost:8501`

La app permite seleccionar cualquiera de los tres modelos desde el sidebar y subir una imagen para obtener la prediccion en tiempo real.

---

## Resultados

| Modelo        | F1-Score | mAP@50 | FPS | Parametros |
|---------------|----------|--------|-----|------------|
| Random Forest | 62.4%    | -      | 9   | -          |
| YOLO11n       | 79.2%    | 77.6%  | 83  | 2.6M       |
| RT-DETR       | 85.1%    | 84.3%  | 28  | 32.0M      |

---

## Dataset

- Nombre: Pothole Detection Dataset YOLOv11 Optimized
- Fuente: https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized
- Total de imagenes: 3,940
- Formato de etiquetas: YOLO (bounding boxes normalizadas)

---

## Repositorio

https://github.com/AlfredoMonT/IABACHES
=======
##Dataset

- **Fuente:** [Pothole Detection Dataset: YOLOv11 Optimized](https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized) — Kaggle
- **Total de imágenes:** 3,940
- **Clases:** `pothole` (bache/rejilla) | `normal` (asfalto sin anomalía)
- **División:** 70% train / 15% val / 15% test

> ⚠️ Las imágenes **no se incluyen** en este repositorio por su tamaño. Descárgalas con el script `scripts/download_dataset.py`.

---

##Fases del Proyecto

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

##Resultados Fase 1

> _Resultados pendientes de experimentación. Se actualizarán al completar el entrenamiento._

| Modelo | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Random Forest + HOG | TBD | TBD | TBD |

---

##Referencias

1. [Pothole Detection and Recognition based on Transfer Learning (arXiv, 2025)](https://arxiv.org/)
2. [Road Surface Condition Detection with Machine Learning using DOT Cameras (arXiv, 2025)](https://arxiv.org/)
3. [Computer Vision-Based Detection and Classification of Road Obstacles (IEEE, 2024)](https://ieeexplore.ieee.org/)
4. [Deep Learning Enhanced feature extraction of Potholes Using Vision and LiDAR (IEEE, 2024)](https://ieeexplore.ieee.org/)
5. [Enhancing Pothole Detection and Characterization: Integrated Segmentation (arXiv, 2025)](https://arxiv.org/)

---

##Licencia

MIT License — ver archivo [LICENSE](LICENSE)

---

*Universidad Andina del Cusco — Escuela Profesional de Ingeniería de Sistemas — 2026*
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
