"""Bayesian Optimisation over a discrete candidate set.

Acquisition policies are formulated for *minimisation* (we negate inside
:func:`compose_objective`). For Expected Improvement we follow the form in
``L48_rc667_Task2.ipynb`` (cell 40), adapted to minimise:

.. math::

    \\mathrm{EI}(x) = (f^* - \\mu(x)) \\Phi(z) + \\sigma(x) \\phi(z),
    \\quad z = (f^* - \\mu(x)) / \\sigma(x)

where :math:`f^*` is the current best (minimum) observed value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.stats import norm

from .gp import GPModel, fit_gp_hyperparameters

logger = logging.getLogger(__name__)


AcquisitionName = Literal["EI", "UCB"]


# ---------------------------------------------------------------------------
# Acquisitions (all formulated for MINIMISATION)
# ---------------------------------------------------------------------------
def expected_improvement(
    f_best: float, mu: np.ndarray, var: np.ndarray, *, xi: float = 0.0
) -> np.ndarray:
    """Expected Improvement for minimisation.

    Args:
        f_best: Current best (minimum) observed ``y``.
        mu: ``(n,)`` posterior mean at candidates.
        var: ``(n,)`` posterior variance at candidates (clipped to >= 0).
        xi: Exploration boost (subtracted from ``f_best``); defaults to 0.

    Returns:
        ``(n,)`` non-negative EI values; argmax = next point.
    """
    sigma = np.sqrt(np.clip(var, 0.0, None))
    improvement = (f_best - xi) - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, improvement / sigma, 0.0)
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma == 0.0] = 0.0
    return np.clip(ei, 0.0, None)


def upper_confidence_bound(
    mu: np.ndarray, var: np.ndarray, *, beta: float = 2.0
) -> np.ndarray:
    """Lower-confidence-bound (we minimise, so smaller LCB is better).

    Returned values are ``-LCB`` so the ``argmax`` convention still gives the
    next point (consistent with EI's "higher is better").
    """
    sigma = np.sqrt(np.clip(var, 0.0, None))
    lcb = mu - beta * sigma
    return -lcb


def acquisition_values(
    name: AcquisitionName,
    mu: np.ndarray,
    var: np.ndarray,
    *,
    f_best: float,
    beta: float = 2.0,
    xi: float = 0.0,
) -> np.ndarray:
    """Dispatch to a registered acquisition (always returns "higher = better")."""
    if name == "EI":
        return expected_improvement(f_best, mu, var, xi=xi)
    if name == "UCB":
        return upper_confidence_bound(mu, var, beta=beta)
    raise ValueError(f"Unknown acquisition {name!r}")


# ---------------------------------------------------------------------------
# Suggest next + BO loop
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Suggestion:
    """One acquisition decision over a candidate set."""

    index: int
    x_next: np.ndarray
    alpha: np.ndarray  # acquisition values across all candidates
    mu: np.ndarray
    var: np.ndarray


def suggest_next(
    model: GPModel,
    x_candidates: np.ndarray,
    y_observed: np.ndarray,
    *,
    acquisition: AcquisitionName = "EI",
    excluded_indices: set[int] | None = None,
    beta: float = 2.0,
    xi: float = 0.0,
) -> Suggestion:
    """Pick the candidate maximising the acquisition.

    Args:
        model: Fitted :class:`GPModel`.
        x_candidates: ``(n_cand, d)`` candidate inputs (z-scored to match
            ``model.x_train``).
        y_observed: Observed targets so far (used to compute ``f_best``).
        acquisition: ``"EI"`` or ``"UCB"``.
        excluded_indices: Optional set of candidate indices to mask out (e.g.
            already-queried geometries during a simulated BO loop).
        beta: UCB exploration coefficient.
        xi: EI exploration boost.

    Returns:
        :class:`Suggestion`.
    """
    if y_observed.size == 0:
        f_best = float("inf")  # Anything looks like an improvement on iter 0
    else:
        f_best = float(np.min(y_observed))

    posterior = model.predict(x_candidates)
    alpha = acquisition_values(
        acquisition, posterior.mean, posterior.var, f_best=f_best, beta=beta, xi=xi
    )

    if excluded_indices:
        mask = np.ones(alpha.shape, dtype=bool)
        excluded = np.array(sorted(excluded_indices), dtype=int)
        excluded = excluded[(excluded >= 0) & (excluded < alpha.size)]
        mask[excluded] = False
        if not np.any(mask):
            raise ValueError("All candidates have been excluded")
        alpha_masked = np.where(mask, alpha, -np.inf)
    else:
        alpha_masked = alpha

    idx = int(np.argmax(alpha_masked))
    return Suggestion(
        index=idx,
        x_next=x_candidates[idx],
        alpha=alpha,
        mu=posterior.mean,
        var=posterior.var,
    )


# ---------------------------------------------------------------------------
# BO loop
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BOHistory:
    """History of a discrete BO run.

    Attributes:
        visited_indices: Indices into ``x_candidates`` queried in order.
        x_visited: Inputs queried, in order. Shape ``(n, d)``.
        y_visited: Observed targets, in order. Shape ``(n,)``.
        best_y_curve: Running minimum of ``y_visited``.
        models: Fitted :class:`GPModel` after each iteration (length
            ``n_iter`` -- the model used to choose iteration ``i+1``'s point
            is at index ``i``).
        suggestions: :class:`Suggestion` from each iteration.
    """

    visited_indices: list[int]
    x_visited: np.ndarray
    y_visited: np.ndarray
    best_y_curve: np.ndarray
    models: list[GPModel]
    suggestions: list[Suggestion]


ObjectiveCallable = Callable[[np.ndarray, int], float]


def run_bo_loop(
    *,
    x_candidates: np.ndarray,
    objective_fn: ObjectiveCallable,
    initial_indices: list[int],
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    extra_train_x: np.ndarray | None = None,
    extra_train_y: np.ndarray | None = None,
    n_iter: int = 10,
    kernel: str = "matern52",
    acquisition: AcquisitionName = "EI",
    n_restarts: int = 3,
    refit_every: int = 1,
    seed: int | None = 0,
    forbid_revisits: bool = True,
) -> BOHistory:
    """Run a discrete BO loop on ``x_candidates``.

    The objective is queried via ``objective_fn(x_chosen, idx)``. Training
    data optionally includes a *prior pool* (``extra_train_x``/``extra_train_y``)
    that augments the BO-acquired observations -- this is how the
    ``pool_with_self`` and ``pool_loo`` modes inject everyone-else's data.

    Args:
        x_candidates: ``(n_cand, d)`` candidate inputs.
        objective_fn: Callable ``(x, idx) -> y`` that returns the observed
            target for the chosen candidate.
        initial_indices: Indices into ``x_candidates`` already evaluated.
        initial_x: Their inputs (``(k, d)``); usually ``x_candidates[initial_indices]``.
        initial_y: Their observed targets (``(k,)``).
        extra_train_x: Optional prior-pool inputs (not in the candidate set).
        extra_train_y: Optional prior-pool targets.
        n_iter: Number of BO iterations after the initial seed.
        kernel: GP kernel name passed to :func:`fit_gp_hyperparameters`.
        acquisition: ``"EI"`` or ``"UCB"``.
        n_restarts: GP fit restart count.
        refit_every: Refit hyperparameters every ``k`` iterations (1 = always).
        seed: Forwarded to the GP fit RNG.
        forbid_revisits: If True (default), already-visited candidate indices
            are masked from the acquisition.

    Returns:
        :class:`BOHistory`.

    Raises:
        ValueError: On shape mismatches.
    """
    if initial_x.ndim != 2 or initial_x.shape[1] != x_candidates.shape[1]:
        raise ValueError("initial_x has wrong shape")
    if initial_y.shape[0] != initial_x.shape[0]:
        raise ValueError("initial_x / initial_y length mismatch")
    if (extra_train_x is None) != (extra_train_y is None):
        raise ValueError("extra_train_x and extra_train_y must be both set or both None")
    if extra_train_x is not None and extra_train_x.shape[1] != x_candidates.shape[1]:
        raise ValueError("extra_train_x has wrong feature dim")

    visited_indices = list(initial_indices)
    x_visited = initial_x.copy()
    y_visited = initial_y.copy()
    best_y_curve: list[float] = []
    if y_visited.size > 0:
        best_y_curve.append(float(np.min(y_visited)))

    extra_x = extra_train_x if extra_train_x is not None else np.zeros((0, x_candidates.shape[1]))
    extra_y = extra_train_y if extra_train_y is not None else np.zeros((0,))

    models: list[GPModel] = []
    suggestions: list[Suggestion] = []
    cached_model: GPModel | None = None

    for it in range(n_iter):
        x_train = np.vstack([extra_x, x_visited]) if x_visited.size else extra_x
        y_train = np.concatenate([extra_y, y_visited]) if y_visited.size else extra_y

        if x_train.shape[0] == 0:
            raise ValueError(
                "BO loop has no training data on iteration 0. "
                "Pass either initial_indices or extra_train_x."
            )

        if cached_model is None or it % max(1, refit_every) == 0:
            cached_model = fit_gp_hyperparameters(
                x_train, y_train, kernel=kernel, n_restarts=n_restarts, seed=seed
            )
        else:
            # reuse theta but refresh training cache
            cached_model = GPModel(
                kernel_name=cached_model.kernel_name,
                theta=cached_model.theta,
                x_train=x_train,
                y_train=y_train,
                train_nll=cached_model.train_nll,
            )

        excluded = set(visited_indices) if forbid_revisits else None
        suggestion = suggest_next(
            cached_model, x_candidates, y_visited, acquisition=acquisition,
            excluded_indices=excluded,
        )

        idx = suggestion.index
        x_next = suggestion.x_next
        y_next = float(objective_fn(x_next, idx))

        visited_indices.append(idx)
        x_visited = np.vstack([x_visited, x_next.reshape(1, -1)])
        y_visited = np.concatenate([y_visited, [y_next]])
        best_y_curve.append(float(np.min(y_visited)))

        models.append(cached_model)
        suggestions.append(suggestion)

    return BOHistory(
        visited_indices=visited_indices,
        x_visited=x_visited,
        y_visited=y_visited,
        best_y_curve=np.asarray(best_y_curve, dtype=float),
        models=models,
        suggestions=suggestions,
    )
