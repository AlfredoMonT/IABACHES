<<<<<<< HEAD
# Arquitectura del Sistema

## Pipeline por Fases

```
Imagen de entrada
      |
      v
Preprocesamiento (resize 640x640, CLAHE, escala de grises)
      |
      |-----> Fase 1: HOG + Random Forest --> Clasificacion (bache / normal)
      |
      |-----> Fase 2: YOLO11n             --> Bounding Boxes (83 FPS)
      |
      |-----> Fase 3: RT-DETR             --> Bounding Boxes (28 FPS, mayor precision)
      |
      v
Interfaz Streamlit (app/app.py)
```

## Comparativa de Modelos

| Modelo        | F1-Score | mAP@50 | FPS | Parametros |
|---------------|----------|--------|-----|------------|
| Random Forest | 62.4%    | -      | 9   | -          |
| YOLO11n       | 79.2%    | 77.6%  | 83  | 2.6M       |
| RT-DETR       | 85.1%    | 84.3%  | 28  | 32.0M      |

## Dataset

- Fuente: Kaggle - pothole-detection-dataset-yolov11-optimized
- Total: 3,940 imagenes
- Formato: YOLO (bounding boxes normalizadas)
- Division: 80% train / 20% val
=======
# Diagrama de Arquitectura del Pipeline — Fase 1

```
DATASET KAGGLE (3,940 imágenes)
         │
         ▼
┌─────────────────────────────┐
│     PREPROCESAMIENTO        │
│  • Resize → 64x64           │
│  • Conversión a Grayscale   │
│  • CLAHE (contraste)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   EXTRACCIÓN DE FEATURES    │
│  • HOG (1,764 features)     │
│  • GLCM (4 features)        │
│  • Vector total: ~1,768     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    SPLIT DEL DATASET        │
│  • Train:  70% (~2,758 img) │
│  • Val:    15% (~591 img)   │
│  • Test:   15% (~591 img)   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     RANDOM FOREST           │
│  • 200 estimadores          │
│  • max_features: sqrt       │
│  • class_weight: balanced   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       EVALUACIÓN            │
│  • F1-Score (weighted)      │
│  • Accuracy                 │
│  • AUC-ROC                  │
│  • Matriz de Confusión      │
└─────────────────────────────┘
```

## Flujo de archivos

| Módulo | Archivo | Entrada | Salida |
|--------|---------|---------|--------|
| Preprocesamiento | `src/preprocessing/image_processor.py` | Imágenes raw | Imágenes 64x64 grises |
| Extracción | `src/features/hog_extractor.py` | Imágenes procesadas | Matriz X (n, 1768) |
| Modelo | `src/models/random_forest_model.py` | Matriz X, vector y | Modelo .pkl |
| Evaluación | `src/evaluation/metrics.py` | Predicciones | Gráficas y métricas |
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
