import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
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
    plt.show()
