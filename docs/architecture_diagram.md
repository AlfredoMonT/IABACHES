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
