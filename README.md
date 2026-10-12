# Usadel BVP Solvers

Numerical solution of the Usadel equations for modelling the Josephson effect.

The project implements an adaptive RK32 Runge–Kutta method and a shooting method for boundary value problems, which are then applied to the Usadel equations.

The main analysis is available in [`assignments/project3.ipynb`](assignments/project3.ipynb), while the numerical implementation is organized under `src/`.

## Run

```bash
uv sync
uv run python scripts/usadel.py
```

## Authors

- Kristoffer Eide
- Frederik Alexander Ådlandsvik
- Samuel Vasshus
- Elias Veines-Jensen

TMA4320 – Introduction to Scientific Computing
