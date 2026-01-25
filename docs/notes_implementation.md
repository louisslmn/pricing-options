# Notes

## Pourquoi pas SciPy ?
Pour éviter des dépendances lourdes, on n’utilise pas `scipy.stats.norm`.
- Dans `black_scholes.py`, on utilise `math.erf` (suffisant car scalaire).
- Dans `monte_carlo.py`, on utilise une approximation polynomiale de N(0,1) vectorisée.

## Couverture delta‑hedging
Convention utilisée :
- on vend 1 option (short)
- on achète Delta actions
- on finance la position via un compte cash rémunéré à r
- on rebalance à chaque pas de temps

Résultat : distribution de P&L à maturité, avec VaR / CVaR.

