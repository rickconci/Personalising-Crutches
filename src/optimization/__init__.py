"""Hand-written GP + Bayesian Optimisation pipeline for crutch personalisation.

Modules:
    * :mod:`data` -- load + join the three JSON sources, build the dense
      candidate grid, z-score helpers, composite objective builder.
    * :mod:`gp` -- numpy/scipy GP (RBF + Matern52, Cholesky posterior, NLL fit).
    * :mod:`bo` -- acquisition functions (EI, UCB) and the discrete BO loop.
    * :mod:`pipeline` -- orchestrator with three personalisation prior modes.
    * :mod:`viz` -- cross-participant visualisations (Plotly).
"""

from __future__ import annotations
