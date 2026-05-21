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
