"""
metrics.py
----------
Funciones de evaluación y visualización de métricas.
Universidad Andina del Cusco — Proyecto IABACHES
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    f1_score, accuracy_score, classification_report
)
from pathlib import Path


CLASES = ['Asfalto Normal', 'Bache/Rejilla']


def reporte_completo(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None):
    """Imprime el reporte completo de clasificación."""
    print('=' * 50)
    print('REPORTE DE EVALUACIÓN — FASE 1')
    print('=' * 50)
    print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')
    print(f'F1-Score : {f1_score(y_true, y_pred, average="weighted"):.4f}')
    if y_proba is not None:
        print(f'AUC-ROC  : {roc_auc_score(y_true, y_proba):.4f}')
    print()
    print(classification_report(y_true, y_pred, target_names=CLASES))


def graficar_confusion_matrix(y_true, y_pred, save_path=None):
    """Genera y guarda la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASES, yticklabels=CLASES)
    plt.title('Matriz de Confusión — Random Forest (Fase 1)', fontweight='bold')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def graficar_roc(y_true, y_proba, save_path=None):
    """Genera y guarda la curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'navy', lw=1, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC — Fase 1', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
