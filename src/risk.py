"""
risk.py
=======

Mesures de risque simples sur une distribution de P&L.
"""

from __future__ import annotations

import numpy as np


def var_cvar(pnl: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    """
    VaR et CVaR (aussi appelé Expected Shortfall).

    Convention :
    - pnl > 0 : gain
    - pnl < 0 : perte

    VaR_alpha = quantile des pertes à (1-alpha)
    Ici on renvoie VaR/CVaR en *perte positive* (plus intuitif).

    Exemple : alpha=0.95 -> VaR 95% : "perte max" à 95% de confiance.
    """
    pnl = np.asarray(pnl, dtype=float)
    q = np.quantile(pnl, 1.0 - alpha)
    pertes = -pnl  # pertes positives
    var = float(np.quantile(pertes, alpha))
    cvar = float(pertes[pertes >= var].mean()) if np.any(pertes >= var) else float(var)
    return var, cvar
