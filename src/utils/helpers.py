"""
helpers.py
----------
Funciones de utilidad general del proyecto.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def verificar_estructura():
    """Verifica que la estructura de carpetas del proyecto esté completa."""
    root = Path(__file__).resolve().parents[2]
    carpetas = [
        'data/raw', 'data/processed', 'data/samples',
        'notebooks', 'src', 'reports/fase1', 'reports/fase2', 'reports/fase3',
        'results/metrics', 'results/figures', 'results/models'
    ]
    print('📁 Verificando estructura del proyecto:')
    for carpeta in carpetas:
        ruta = root / carpeta
        estado = '✅' if ruta.exists() else '❌'
        print(f'   {estado} {carpeta}')


def mostrar_muestra_imagenes(imagenes: list, titulos: list = None, cols: int = 3):
    """Visualiza una grilla de imágenes."""
    rows = (len(imagenes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i, (ax, img) in enumerate(zip(axes, imagenes)):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        if titulos and i < len(titulos):
            ax.set_title(titulos[i], fontsize=9)
        ax.axis('off')

    for ax in axes[len(imagenes):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
