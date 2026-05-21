"""
random_forest_model.py
-----------------------
Entrenamiento, guardado y carga del modelo Random Forest para Fase 1.
Universidad Andina del Cusco — Proyecto IABACHES
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


RANDOM_SEED = 42


def crear_modelo(n_estimators: int = 200) -> RandomForestClassifier:
    """
    Crea una instancia del modelo Random Forest con los hiperparámetros del proyecto.

    Args:
        n_estimators: Número de árboles en el bosque.

    Returns:
        Instancia de RandomForestClassifier configurada.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def entrenar(modelo: RandomForestClassifier, X_train: np.ndarray, y_train: np.ndarray):
    """Entrena el modelo con los datos de entrenamiento."""
    modelo.fit(X_train, y_train)
    return modelo


def evaluar(modelo: RandomForestClassifier, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evalúa el modelo y retorna un diccionario de métricas.

    Returns:
        Dict con accuracy, f1_score y auc_roc.
    """
    y_pred  = modelo.predict(X)
    y_proba = modelo.predict_proba(X)[:, 1]

    return {
        'accuracy': round(accuracy_score(y, y_pred), 4),
        'f1_score': round(f1_score(y, y_pred, average='weighted'), 4),
        'auc_roc':  round(roc_auc_score(y, y_proba), 4),
    }


def guardar_modelo(modelo: RandomForestClassifier, ruta: str | Path):
    """Guarda el modelo entrenado en disco."""
    joblib.dump(modelo, str(ruta))
    print(f'✅ Modelo guardado: {ruta}')


def cargar_modelo(ruta: str | Path) -> RandomForestClassifier:
    """Carga un modelo previamente entrenado."""
    return joblib.load(str(ruta))
