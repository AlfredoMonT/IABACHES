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
```

---

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
```

---

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
