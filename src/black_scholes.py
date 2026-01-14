"""
black_scholes.py
================

Implémentation du modèle de Black‑Scholes (options européennes).
On évite SciPy pour garder les dépendances légères.

Rappels (sans dividendes) :
- d1 = (ln(S/K) + (r + 0.5 sigma^2) T) / (sigma sqrt(T))
- d2 = d1 - sigma sqrt(T)

Call :
  C = S N(d1) - K e^{-rT} N(d2)

Put :
  P = K e^{-rT} N(-d2) - S N(-d1)

Greeks :
- Delta, Gamma, Vega, Theta, Rho
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


def _norm_cdf(x: float) -> float:
    """CDF d'une N(0,1) via erf (standard library)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """PDF d'une N(0,1)."""
    return (1.0 / sqrt(2.0 * 3.141592653589793)) * exp(-0.5 * x * x)


@dataclass(frozen=True)
class BSInputs:
    S: float
    K: float
    T: float  # maturité en années
    r: float  # taux sans risque (continu)
    sigma: float  # volatilité annualisée


def _d1_d2(x: BSInputs) -> tuple[float, float]:
    if x.S <= 0 or x.K <= 0:
        raise ValueError("S et K doivent être > 0.")
    if x.T <= 0:
        raise ValueError("T doit être > 0.")
    if x.sigma <= 0:
        raise ValueError("sigma doit être > 0.")

    d1 = (log(x.S / x.K) + (x.r + 0.5 * x.sigma**2) * x.T) / (x.sigma * sqrt(x.T))
    d2 = d1 - x.sigma * sqrt(x.T)
    return d1, d2


def prix_bs(x: BSInputs, option_type: OptionType = "call") -> float:
    """Prix Black‑Scholes d'une option européenne (sans dividendes)."""
    d1, d2 = _d1_d2(x)
    df = exp(-x.r * x.T)  # discount factor

    if option_type == "call":
        return x.S * _norm_cdf(d1) - x.K * df * _norm_cdf(d2)
    elif option_type == "put":
        return x.K * df * _norm_cdf(-d2) - x.S * _norm_cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def greeks_bs(x: BSInputs, option_type: OptionType = "call") -> Greeks:
    """
    Greeks Black‑Scholes (sans dividendes).

    Notes :
    - Vega est donné par 1 point de volatilité (ex: 0.01), donc on renvoie par unité de sigma.
    - Theta est annuel (car T est en années). On le renvoie "par an".
    """
    d1, d2 = _d1_d2(x)
    pdf = _norm_pdf(d1)
    df = exp(-x.r * x.T)

    gamma = pdf / (x.S * x.sigma * sqrt(x.T))
    vega = x.S * pdf * sqrt(x.T)

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = -(x.S * pdf * x.sigma) / (2.0 * sqrt(x.T)) - x.r * x.K * df * _norm_cdf(d2)
        rho = x.K * x.T * df * _norm_cdf(d2)
    elif option_type == "put":
        delta = _norm_cdf(d1) - 1.0
        theta = -(x.S * pdf * x.sigma) / (2.0 * sqrt(x.T)) + x.r * x.K * df * _norm_cdf(-d2)
        rho = -x.K * x.T * df * _norm_cdf(-d2)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    return Greeks(delta=float(delta), gamma=float(gamma), vega=float(vega), theta=float(theta), rho=float(rho))


def parite_put_call(x: BSInputs) -> float:
    """
    Vérifie la parité put-call :
        C - P = S - K e^{-rT}
    Retourne la partie droite (forward) : S - K e^{-rT}
    """
    return x.S - x.K * exp(-x.r * x.T)
