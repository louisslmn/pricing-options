import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from black_scholes import BSInputs, prix_bs, greeks_bs, parite_put_call
from monte_carlo import MCConfig, prix_option_mc, HedgeConfig, simuler_delta_hedging
from risk import var_cvar
from visualisation import fig_sensibilite, fig_histogramme


st.set_page_config(page_title="Options", page_icon="🧮", layout="wide")

st.title("Pricing d’options & couverture (Black‑Scholes + Monte Carlo)")
st.caption(
    "Objectif : montrer des compétences Python (calcul numérique, simulation, visualisation) "
    "sur un sujet finance de marché très classique."
)

# ----------------------------
# Sidebar : paramètres généraux
# ----------------------------
st.sidebar.header("Paramètres option")

option_type = st.sidebar.selectbox("Type d'option", ["call", "put"], format_func=lambda x: "Call" if x=="call" else "Put")
S = st.sidebar.number_input("Prix spot S", min_value=1.0, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike K", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Maturité T (années)", min_value=0.01, value=1.0, step=0.05)
r = st.sidebar.number_input("Taux sans risque r (continu)", min_value=-0.05, value=0.02, step=0.005, format="%.3f")
sigma = st.sidebar.number_input("Volatilité σ", min_value=0.01, value=0.20, step=0.01, format="%.2f")

x = BSInputs(S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma))

tab_pricing, tab_hedge, tab_explain = st.tabs(["Pricing & Greeks", "Couverture (delta-hedging)", "Explications"])


# ============================
# TAB 1 — Pricing & Greeks
# ============================
with tab_pricing:
    st.subheader("Prix Black‑Scholes (analytique)")
    prix_analytique = prix_bs(x, option_type=option_type)
    g = greeks_bs(x, option_type=option_type)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Prix", f"{prix_analytique:,.4f}")
    c2.metric("Delta", f"{g.delta:,.4f}")
    c3.metric("Gamma", f"{g.gamma:,.6f}")
    c4.metric("Vega", f"{g.vega:,.4f}")
    c5.metric("Theta (par an)", f"{g.theta:,.4f}")
    c6.metric("Rho", f"{g.rho:,.4f}")

    with st.expander("Check rapide : parité put‑call", expanded=False):
        rhs = parite_put_call(x)
        st.markdown(
            f"""
La parité put‑call (sans dividendes) dit :

- **C − P = S − K e^(−rT)**  
Ici, **S − K e^(−rT) = {rhs:,.4f}**

Si vous calculez Call et Put avec les mêmes paramètres, la différence doit être proche de ce nombre.
            """
        )

    st.divider()

    st.subheader("Prix Monte Carlo (comparaison)")
    colA, colB = st.columns([0.6, 0.4])
    with colA:
        n_paths = st.slider("Nombre de trajectoires", 2_000, 200_000, 50_000, 2_000)
        n_steps = st.slider("Nombre de pas de temps", 50, 500, 252, 10)
        seed = st.number_input("Seed", value=42, step=1)

        cfg = MCConfig(n_paths=int(n_paths), n_steps=int(n_steps), seed=int(seed))
        prix_mc, se = prix_option_mc(x, option_type=option_type, cfg=cfg)

        st.write(f"Prix MC : **{prix_mc:,.4f}** (erreur standard ≈ **{se:,.4f}**)")
        st.write(f"Écart vs analytique : **{(prix_mc - prix_analytique):+.4f}**")

    with colB:
        st.info(
            "En augmentant `n_paths`, l'erreur standard diminue ~ en 1/√n.\n\n"
            "Le prix analytique est une référence (dans le cadre du modèle)."
        )

    st.divider()

    st.subheader("Sensibilités (intuition visuelle)")
    S_grid = np.linspace(max(1.0, 0.5 * S), 1.5 * S, 60)
    prix_grid = []
    delta_grid = []
    for s_val in S_grid:
        xi = BSInputs(S=float(s_val), K=x.K, T=x.T, r=x.r, sigma=x.sigma)
        prix_grid.append(prix_bs(xi, option_type=option_type))
        delta_grid.append(greeks_bs(xi, option_type=option_type).delta)

    st.pyplot(fig_sensibilite(S_grid, np.array(prix_grid), "Prix vs Spot", "S", "Prix"))
    st.pyplot(fig_sensibilite(S_grid, np.array(delta_grid), "Delta vs Spot", "S", "Delta"))


# ============================
# TAB 2 — Delta hedging
# ============================
with tab_hedge:
    st.subheader("Simulation de delta‑hedging (position short option)")

    st.markdown(
        """
On simule une dynamique GBM pour le sous‑jacent (la “vraie” volatilité est `σ vraie`).
Ensuite on couvre en utilisant une volatilité de modèle `σ modèle` (qui peut être différente).

Cela montre :
- pourquoi la couverture est imparfaite si le modèle est faux,
- l'effet des coûts de transaction.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_paths_h = st.slider("Trajectoires (hedge)", 1_000, 50_000, 10_000, 1_000)
        n_steps_h = st.slider("Pas (hedge)", 50, 500, 252, 10)
        seed_h = st.number_input("Seed (hedge)", value=7, step=1)

    with col2:
        sigma_modele = st.number_input("σ modèle (hedge)", min_value=0.01, value=float(sigma), step=0.01, format="%.2f")
        cout_bps = st.slider("Coût de transaction (bps)", 0.0, 50.0, 0.0, 1.0)

    with col3:
        st.info(
            "Interprétation rapide :\n"
            "- si σ modèle = σ vraie et coût=0 → P&L proche de 0\n"
            "- si σ modèle ≠ σ vraie → dispersion du P&L\n"
            "- si coût > 0 → P&L moyen devient négatif (friction)"
        )

    if st.button("▶️ Lancer la simulation", type="primary"):
        cfg = HedgeConfig(
            n_paths=int(n_paths_h),
            n_steps=int(n_steps_h),
            seed=int(seed_h),
            cout_transaction_bps=float(cout_bps),
            sigma_modele=float(sigma_modele),
        )
        res = simuler_delta_hedging(x, option_type=option_type, cfg=cfg)
        pnl = res["pnl"]

        st.pyplot(fig_histogramme(pnl, bins=60, titre="Distribution du P&L (delta‑hedging)"))

        var95, cvar95 = var_cvar(pnl, alpha=0.95)

        a, b, c, d = st.columns(4)
        a.metric("P&L moyen", f"{pnl.mean():,.4f}")
        b.metric("Écart‑type", f"{pnl.std(ddof=1):,.4f}")
        c.metric("VaR 95% (perte)", f"{var95:,.4f}")
        d.metric("CVaR 95% (perte)", f"{cvar95:,.4f}")

        with st.expander("Voir stats détaillées"):
            st.json(res["stats"])


# ============================
# TAB 3 — Explications
# ============================
with tab_explain:
    st.subheader("Explications (formules + choix d’implémentation)")

    st.markdown(
        """
### Pourquoi Black‑Scholes ?
Black‑Scholes est un standard : une formule fermée existe pour les options européennes.
Cela permet :
- d'avoir un prix de référence
- de calculer des Greeks (sensibilités), utiles pour la gestion des risques

### Pourquoi du Monte Carlo ?
Monte Carlo sert à :
- vérifier numériquement le prix
- traiter des cas où la formule fermée n'existe pas (options exotiques, payoffs complexes, etc.)

### Couverture (delta‑hedging)
L'idée : se protéger des petits mouvements du sous‑jacent en ajustant en continu (ou quasi) une position en action.

En pratique :
- la couverture est discrète (on ne rebalance pas en continu)
- les paramètres (volatilité, taux) sont estimés
- il existe des coûts de transaction

D'où un P&L non nul même si le modèle est “bon”.
        """
    )

    st.markdown(
        """
### Ce que l’application met en œuvre côté Python
- calcul numérique (formules, stabilité)
- simulation vectorisée (NumPy)
- visualisation (Matplotlib)
- structuration d’un mini‑projet (modules `src/`)
- interface Streamlit claire et testable
        """
    )
