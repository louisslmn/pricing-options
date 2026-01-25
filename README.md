# Options : pricing, Greeks & couverture delta 

Cette application permet de :
1) calculer le prix Black‑Scholes d’options européennes (call/put)  
2) calculer les Greeks (Delta, Gamma, Vega, Theta, Rho)  
3) comparer avec un pricing Monte Carlo (GBM)  
4) simuler une couverture delta‑hedging et analyser la distribution du P&L  
5) estimer VaR / CVaR sur la distribution simulée

---

## Lancer l’application

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Remarques de modélisation

- Modèle Black‑Scholes / GBM sans dividendes (choix volontaire pour rester lisible).
- La couverture delta‑hedging est simulée avec un rebalancement discret : même avec un modèle “parfait”, le P&L n’est pas forcément nul (effet discretisation + non‑linéarités).

---

## Structure

- `src/black_scholes.py` : prix + Greeks (sans SciPy)
- `src/monte_carlo.py` : simulation GBM, pricing Monte Carlo, delta‑hedging (vectorisé)
- `src/risk.py` : VaR / CVaR
- `src/visualisation.py` : graphiques Matplotlib
- `app.py` : interface Streamlit

