<<<<<<< HEAD
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


DEFAULT_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


def build_model(params=None):
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)
    return RandomForestClassifier(**p)


def train(X, y, test_size=0.20, params=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = build_model(params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X), model.predict_proba(X)
=======
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
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
