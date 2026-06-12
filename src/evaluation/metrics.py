<<<<<<< HEAD
import json
=======
"""
metrics.py
----------
Funciones de evaluación y visualización de métricas.
Universidad Andina del Cusco — Proyecto IABACHES
"""

>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
<<<<<<< HEAD
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
    }


def print_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=["Asfalto Normal", "Bache/Rejilla"]))


def save_metrics(metrics_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Asfalto Normal", "Bache/Rejilla"],
        yticklabels=["Asfalto Normal", "Bache/Rejilla"],
        linewidths=1,
        linecolor="white",
    )
    ax.set_ylabel("Clase Real")
    ax.set_xlabel("Clase Predicha")
    ax.set_title("Matriz de Confusion")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def plot_metrics_bar(metrics_dict, save_path=None):
    names = ["Precision", "Recall", "F1-Score", "AUC-ROC"]
    values = [metrics_dict["precision"], metrics_dict["recall"],
               metrics_dict["f1"], metrics_dict["auc_roc"]]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valor")
    ax.set_title("Metricas de Evaluacion")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.6, label="Umbral 80%")
    ax.legend()
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
=======
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
>>>>>>> 0a4738871d3f5c6268a7eae1f539c02aa4e80d5b
    plt.show()
