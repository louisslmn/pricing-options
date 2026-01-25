"""
monte_carlo.py
==============

Monte Carlo pour :
- simulation de trajectoires GBM (Geometric Brownian Motion)
- pricing d'option européenne
- simulation de delta-hedging (couverture dynamique)

Objectif : rendre la mécanique claire et expérimentable.

Note dépendances :
- On évite SciPy pour garder les dépendances légères.
- Pour la CDF de la normale standard (N(d1)), on utilise une **approximation polynomiale**
  vectorisée (rapide en NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from black_scholes import BSInputs, prix_bs, greeks_bs


OptionType = Literal["call", "put"]


def norm_cdf_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximation vectorisée de la CDF N(0,1) (Abramowitz & Stegun).
    Précision largement suffisante pour une démo / projet étudiant.

    Référence classique : approximation rationnelle utilisée en pratique.
    """
    x = np.asarray(x, dtype=float)
    sign = np.where(x >= 0, 1.0, -1.0)
    z = np.abs(x)

    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989423 * np.exp(-0.5 * z * z)

    # polynôme
    p = d * t * (
        0.3193815
        + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))
    )

    cdf = np.where(sign > 0, 1.0 - p, p)
    return cdf


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 50_000
    n_steps: int = 252
    seed: int = 42


def simuler_gbm(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Simule des trajectoires GBM :
        dS/S = r dt + sigma dW

    Retour : array shape = (n_paths, n_steps+1)
    """
    if S0 <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S0, T et sigma doivent être > 0.")
    if n_paths <= 0 or n_steps <= 1:
        raise ValueError("n_paths > 0 et n_steps > 1 requis.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Incréments browniens
    Z = rng.standard_normal(size=(n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    logS = np.log(S0) + np.cumsum(increments, axis=1)
    logS = np.concatenate([np.full((n_paths, 1), np.log(S0)), logS], axis=1)
    S = np.exp(logS)
    return S


def prix_option_mc(
    x: BSInputs,
    option_type: OptionType,
    cfg: MCConfig,
) -> Tuple[float, float]:
    """
    Prix d'une option européenne par Monte Carlo + erreur standard.

    On simule S_T puis payoff puis discount.

    Retour : (prix_estime, erreur_standard)
    """
    paths = simuler_gbm(
        S0=x.S,
        T=x.T,
        r=x.r,
        sigma=x.sigma,
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
    )
    ST = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(ST - x.K, 0.0)
    else:
        payoff = np.maximum(x.K - ST, 0.0)

    disc = np.exp(-x.r * x.T)
    prix = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(cfg.n_paths)

    return float(prix), float(se)


@dataclass(frozen=True)
class HedgeConfig:
    n_paths: int = 10_000
    n_steps: int = 252
    seed: int = 7
    cout_transaction_bps: float = 0.0  # coût par rebal (bps du notionnel)
    sigma_modele: float | None = None  # si None, on hedge avec la vraie sigma


def simuler_delta_hedging(
    x: BSInputs,
    option_type: OptionType,
    cfg: HedgeConfig,
) -> dict:
    """
    Simulation de delta hedging d'une option européenne.

    Convention simplifiée :
    - On vend 1 option (short), on hedge en achetant Delta actions.
    - On rebalance à chaque pas de temps.
    - Le cash est rémunéré au taux r (continu).
    - Coût de transaction : proportionnel au *notionnel* échangé à chaque rebal.

    Retourne :
    - pnl : array (n_paths,)
    - stats : quelques statistiques de P&L
    """
    sigma_hedge = cfg.sigma_modele if cfg.sigma_modele is not None else x.sigma

    paths = simuler_gbm(
        S0=x.S,
        T=x.T,
        r=x.r,
        sigma=x.sigma,  # vraie dynamique
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        seed=cfg.seed,
    )

    dt = x.T / cfg.n_steps
    growth = np.exp(x.r * dt)

    # Prix initial (modèle) et delta initial
    x0 = BSInputs(S=x.S, K=x.K, T=x.T, r=x.r, sigma=sigma_hedge)
    prix0 = prix_bs(x0, option_type=option_type)
    delta0 = greeks_bs(x0, option_type=option_type).delta

    # Position initiale : short option -> on reçoit prix0 en cash
    cash = np.full(cfg.n_paths, prix0, dtype=float)

    # On achète delta0 actions pour couvrir
    position_actions = np.full(cfg.n_paths, delta0, dtype=float)
    cash -= position_actions * paths[:, 0]

    # Coût de transaction initial (optionnel)
    if cfg.cout_transaction_bps > 0:
        cash -= (np.abs(position_actions) * paths[:, 0]) * (cfg.cout_transaction_bps / 10_000)

    # Rebalancement
    for t in range(1, cfg.n_steps + 1):
        S_t = paths[:, t]
        T_restante = max(x.T - t * dt, 1e-12)

        # intérêts sur le cash
        cash *= growth

        # delta "modèle" à date t (vectorisé)
        sigma = sigma_hedge
        d1 = (np.log(S_t / x.K) + (x.r + 0.5 * sigma**2) * T_restante) / (sigma * np.sqrt(T_restante))
        Nd1 = norm_cdf_approx(d1)
        delta_t = Nd1 if option_type == "call" else (Nd1 - 1.0)

        # Variation de position (trade)
        trade = delta_t - position_actions
        cash -= trade * S_t

        # Coût de transaction (bps du notionnel échangé)
        if cfg.cout_transaction_bps > 0:
            cash -= (np.abs(trade) * S_t) * (cfg.cout_transaction_bps / 10_000)

        position_actions = delta_t

    # À maturité : on rachète l'option (payoff) et on liquide les actions
    ST = paths[:, -1]
    payoff = np.maximum(ST - x.K, 0.0) if option_type == "call" else np.maximum(x.K - ST, 0.0)

    # Portefeuille final = cash + position_actions*ST - payoff (car on est short option)
    valeur_finale = cash + position_actions * ST - payoff
    pnl = valeur_finale

    stats = {
        "pnl_moyen": float(pnl.mean()),
        "pnl_std": float(pnl.std(ddof=1)),
        "pnl_p5": float(np.quantile(pnl, 0.05)),
        "pnl_p95": float(np.quantile(pnl, 0.95)),
    }
    return {"pnl": pnl, "stats": stats}
