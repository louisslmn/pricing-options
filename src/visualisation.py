"""
visualisation.py
================

Matplotlib uniquement, pour rester léger.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def fig_sensibilite(x: np.ndarray, y: np.ndarray, titre: str, xlabel: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2.0)
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_histogramme(pnl: np.ndarray, bins: int = 60, titre: str = "Distribution du P&L") -> plt.Figure:
    fig, ax = plt.subplots()
    ax.hist(pnl, bins=bins)
    ax.set_title(titre)
    ax.set_xlabel("P&L")
    ax.set_ylabel("Fréquence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
