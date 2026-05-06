"""End-to-end orchestration of GP+BO personalisation.

Three "prior" modes are supported (one function, dispatched on the enum):

1. :attr:`PriorMode.POOL_INCLUDING_SELF` -- use everyone's data, including the
   target participant's own grid-search trials, as the GP training set.
2. :attr:`PriorMode.POOL_LEAVE_ONE_OUT` -- use everyone's data *except* the
   target's, so we measure how well others' data alone personalises a new
   participant.
3. :attr:`PriorMode.NO_PRIOR` -- only the target's data (seeded with
   ``seed_n`` random rows) drives the BO.

For *simulated* runs (``simulate_bo``) the candidate set is restricted to the
geometries actually measured for the target participant, so each "query" returns
a real measured target value. For *live* suggestions (``suggest_next_for``) the
candidate set is the dense 140-combo grid built by
:func:`data.build_candidate_grid`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .bo import AcquisitionName, BOHistory, Suggestion, run_bo_loop, suggest_next
from .data import (
    GEOMETRY_FEATURES,
    INPUT_FEATURES,
    PARTICIPANT_FEATURES,
    ObjectiveBundle,
    ObjectiveKind,
    ZScoreStats,
    assemble_candidate_matrix,
    assemble_design_matrix,
    build_candidate_grid,
    compose_objective,
    fit_zscore,
    make_training_bundle,
)
from .gp import GPModel, KernelName, fit_gp_hyperparameters

logger = logging.getLogger(__name__)


class PriorMode(str, Enum):
    """Prior-pool selection strategy for the GP training set."""

    POOL_INCLUDING_SELF = "pool_with_self"
    POOL_LEAVE_ONE_OUT = "pool_loo"
    NO_PRIOR = "no_prior"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_pool(
    df: pd.DataFrame, *, target_mih: str, mode: PriorMode
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into ``(prior_pool, target_rows)`` per mode.

    ``prior_pool`` is what feeds the GP as fixed prior data; ``target_rows``
    are the candidate measurements for the BO loop on the target participant.
    """
    if target_mih not in set(df["participant"]):
        raise KeyError(f"target_mih={target_mih!r} not in dataset")

    target = df[df["participant"] == target_mih].copy().reset_index(drop=True)
    others = df[df["participant"] != target_mih].copy().reset_index(drop=True)

    if mode is PriorMode.POOL_INCLUDING_SELF:
        prior = pd.concat([others, target], ignore_index=True)
    elif mode is PriorMode.POOL_LEAVE_ONE_OUT:
        prior = others
    elif mode is PriorMode.NO_PRIOR:
        prior = others.iloc[0:0].copy()  # empty, but with same columns
    else:
        raise ValueError(f"Unknown PriorMode: {mode}")

    return prior, target


def _participant_features(df: pd.DataFrame, target_mih: str) -> pd.Series:
    """Return one row of participant features for ``target_mih``."""
    rows = df[df["participant"] == target_mih]
    if rows.empty:
        raise KeyError(f"No rows for participant {target_mih}")
    return rows[list(PARTICIPANT_FEATURES)].iloc[0]


# ---------------------------------------------------------------------------
# Live suggestion: "what should we test next?"
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LiveSuggestion:
    """Single next-geometry suggestion for a target participant.

    Attributes:
        next_alpha, next_beta, next_gamma: Suggested geometry.
        acquisition: Name of the policy used.
        acquisition_value: ``alpha`` value at the chosen point.
        posterior_mean: Posterior mean across all 140 candidates.
        posterior_var: Posterior variance across all 140 candidates.
        candidate_grid: The DataFrame of candidate inputs (140 rows).
        model: The fitted :class:`GPModel`.
    """

    next_alpha: float
    next_beta: float
    next_gamma: float
    acquisition: AcquisitionName
    acquisition_value: float
    posterior_mean: np.ndarray
    posterior_var: np.ndarray
    candidate_grid: pd.DataFrame
    model: GPModel


def suggest_next_for(
    target_mih: str,
    *,
    df: pd.DataFrame,
    mode: PriorMode,
    objective: ObjectiveKind = "combined",
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    candidate_grid: pd.DataFrame | None = None,
    kernel: KernelName = "matern52",
    acquisition: AcquisitionName = "EI",
    n_restarts: int = 3,
    seed: int | None = 0,
    extra_observed_indices: Iterable[int] | None = None,
) -> LiveSuggestion:
    """Recommend the next ``(alpha, beta, gamma)`` for ``target_mih``.

    Trains a GP on the prior pool (selected by ``mode``) using the chosen
    objective, then maximises the acquisition over the dense 140-combo
    candidate grid (with the target's participant features broadcast).

    Args:
        target_mih: e.g. ``"MIH19"``.
        df: Output of :func:`data.load_joined_dataset`.
        mode: :class:`PriorMode`.
        objective: ``"cot"`` / ``"survey"`` / ``"combined"``.
        w_cot, w_survey: Weights for the combined objective.
        candidate_grid: Override the dense 140-combo grid (useful for tests).
        kernel: GP kernel.
        acquisition: ``"EI"`` or ``"UCB"``.
        n_restarts: Hyperparameter restarts.
        seed: GP fit seed.
        extra_observed_indices: Mask out these candidate-grid rows (e.g.
            geometries already tested live).

    Returns:
        :class:`LiveSuggestion`.
    """
    if candidate_grid is None:
        candidate_grid = build_candidate_grid()

    prior_df, _ = _split_pool(df, target_mih=target_mih, mode=mode)
    if prior_df.empty:
        raise ValueError(
            f"Mode {mode.value} yields an empty prior pool. Use "
            "simulate_bo with seed_n>=1, or supply some target observations."
        )

    bundle = make_training_bundle(
        prior_df, objective=objective, w_cot=w_cot, w_survey=w_survey
    )
    model = fit_gp_hyperparameters(
        bundle.X_train, bundle.y_train, kernel=kernel, n_restarts=n_restarts, seed=seed
    )

    target_features = _participant_features(df, target_mih)
    X_cand = assemble_candidate_matrix(
        candidate_grid, target_features, feature_stats=bundle.feature_stats
    )

    suggestion = suggest_next(
        model, X_cand, bundle.y_train,
        acquisition=acquisition,
        excluded_indices=set(extra_observed_indices or []),
    )

    row = candidate_grid.iloc[suggestion.index]
    return LiveSuggestion(
        next_alpha=float(row["alpha"]),
        next_beta=float(row["beta"]),
        next_gamma=float(row["gamma"]),
        acquisition=acquisition,
        acquisition_value=float(suggestion.alpha[suggestion.index]),
        posterior_mean=suggestion.mu,
        posterior_var=suggestion.var,
        candidate_grid=candidate_grid.copy(),
        model=model,
    )


# ---------------------------------------------------------------------------
# Simulated BO: replay the target's measured geometries via BO
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BOResult:
    """Outcome of a simulated BO run on a target participant.

    Attributes:
        target_mih: Target participant id.
        mode: Prior pool selection strategy used.
        objective: Objective kind used.
        history: BO loop history.
        target_geometries: DataFrame of the target's candidate geometries
            (the simulated candidate set), with their measured target ``y``.
        best_geometry: The (alpha, beta, gamma) of the best ``y`` found.
        best_y: The best ``y`` value (lower = better).
    """

    target_mih: str
    mode: PriorMode
    objective: ObjectiveKind
    history: BOHistory
    target_geometries: pd.DataFrame
    best_geometry: tuple[float, float, float]
    best_y: float


def _build_target_candidates(
    target_df: pd.DataFrame,
    *,
    objective: ObjectiveKind,
    output_stats: ZScoreStats,
    feature_stats: ZScoreStats,
    w_cot: float,
    w_survey: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Construct the simulated-BO candidate set for the target participant.

    Drops rows missing the columns required by the objective, then computes
    the per-row ``y`` using the **prior-pool's** z-stats (so prior + target
    targets are commensurate when evaluated on the same scale).
    """
    needed = ["cot_linear"] if objective == "cot" else (
        ["sus", "nrs", "tlx"] if objective == "survey" else
        ["cot_linear", "sus", "nrs", "tlx"]
    )
    df = target_df.dropna(subset=list(INPUT_FEATURES) + needed).reset_index(drop=True)
    if df.empty:
        raise ValueError("Target participant has no rows with all required outputs")

    y = compose_objective(
        df, objective=objective, stats=output_stats, w_cot=w_cot, w_survey=w_survey
    ).to_numpy(dtype=float)
    X = assemble_design_matrix(df, feature_stats=feature_stats)
    return df, X, y


def simulate_bo(
    target_mih: str,
    *,
    df: pd.DataFrame,
    mode: PriorMode,
    objective: ObjectiveKind = "combined",
    n_iter: int = 8,
    seed_n: int = 1,
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    kernel: KernelName = "matern52",
    acquisition: AcquisitionName = "EI",
    n_restarts: int = 3,
    refit_every: int = 1,
    seed: int | None = 0,
) -> BOResult:
    """Replay BO on the target's measured geometries.

    The candidate set is the *target's* recorded trials (so each "query" is a
    real measurement). Z-stats for both inputs and outputs are fitted on
    whichever rows the GP would actually see at iteration 0:

    * ``POOL_INCLUDING_SELF`` / ``POOL_LEAVE_ONE_OUT``: z-stats fit on the
      prior pool (others ± self).
    * ``NO_PRIOR``: z-stats fit on the target's own rows only (so this mode
      is genuinely cold-start).

    Args:
        target_mih: Target participant id.
        df: Output of :func:`data.load_joined_dataset`.
        mode: :class:`PriorMode`.
        objective: Objective kind.
        n_iter: BO iterations after the seed.
        seed_n: Number of random target rows used to seed the BO. Ignored for
            ``POOL_INCLUDING_SELF`` (uses 0 -- the GP already has self data).
        w_cot, w_survey: Weights for the combined objective.
        kernel, acquisition, n_restarts, refit_every: Forwarded to
            :func:`bo.run_bo_loop`.
        seed: RNG seed for the seed-row sampling and GP fits.

    Returns:
        :class:`BOResult`.
    """
    prior_df, target_df = _split_pool(df, target_mih=target_mih, mode=mode)

    # Z-stat fitting -------------------------------------------------------
    if mode is PriorMode.NO_PRIOR:
        ref_df = target_df.dropna(subset=list(INPUT_FEATURES))
    else:
        ref_df = prior_df.dropna(subset=list(INPUT_FEATURES))

    if ref_df.empty:
        raise ValueError("Reference DataFrame for z-stats is empty")

    feature_stats = fit_zscore(ref_df, INPUT_FEATURES)
    output_stats = fit_zscore(ref_df, ["cot_linear", "sus", "nrs", "tlx"])

    # Candidate set (target's measured geometries) -------------------------
    candidate_df, X_cand, y_cand = _build_target_candidates(
        target_df, objective=objective, output_stats=output_stats,
        feature_stats=feature_stats, w_cot=w_cot, w_survey=w_survey,
    )

    # Prior-pool training matrices -----------------------------------------
    if not prior_df.empty:
        prior_bundle = _bundle_with_stats(
            prior_df, objective=objective, feature_stats=feature_stats,
            output_stats=output_stats, w_cot=w_cot, w_survey=w_survey,
        )
        extra_x: np.ndarray | None = prior_bundle.X_train
        extra_y: np.ndarray | None = prior_bundle.y_train
    else:
        extra_x = None
        extra_y = None

    # Initial seed for the target's own observations -----------------------
    rng = np.random.default_rng(seed)
    if mode is PriorMode.POOL_INCLUDING_SELF:
        # GP already has self in the prior pool; no need to seed BO
        init_indices: list[int] = []
        init_x = np.zeros((0, X_cand.shape[1]))
        init_y = np.zeros((0,))
    else:
        k = max(0, min(int(seed_n), len(candidate_df)))
        if k == 0 and (extra_x is None or extra_x.shape[0] == 0):
            raise ValueError(
                "NO_PRIOR with seed_n=0 has no training data on iter 0; "
                "set seed_n>=1 or use a pooled mode."
            )
        chosen = rng.choice(len(candidate_df), size=k, replace=False) if k > 0 else np.array([], dtype=int)
        init_indices = sorted(int(i) for i in chosen)
        init_x = X_cand[init_indices]
        init_y = y_cand[init_indices]

    def objective_fn(x_chosen: np.ndarray, idx: int) -> float:
        return float(y_cand[idx])

    history = run_bo_loop(
        x_candidates=X_cand,
        objective_fn=objective_fn,
        initial_indices=init_indices,
        initial_x=init_x,
        initial_y=init_y,
        extra_train_x=extra_x,
        extra_train_y=extra_y,
        n_iter=n_iter,
        kernel=kernel,
        acquisition=acquisition,
        n_restarts=n_restarts,
        refit_every=refit_every,
        seed=seed,
        forbid_revisits=True,
    )

    best_idx = int(history.visited_indices[int(np.argmin(history.y_visited))])
    best_row = candidate_df.iloc[best_idx]
    target_geo = candidate_df.assign(y_objective=y_cand)

    return BOResult(
        target_mih=target_mih,
        mode=mode,
        objective=objective,
        history=history,
        target_geometries=target_geo[
            ["participant", "trial_key", *GEOMETRY_FEATURES, "cot_linear", "sus", "nrs", "tlx", "y_objective"]
        ].copy(),
        best_geometry=(float(best_row["alpha"]), float(best_row["beta"]), float(best_row["gamma"])),
        best_y=float(history.y_visited.min()),
    )


def _bundle_with_stats(
    df: pd.DataFrame,
    *,
    objective: ObjectiveKind,
    feature_stats: ZScoreStats,
    output_stats: ZScoreStats,
    w_cot: float,
    w_survey: float,
) -> ObjectiveBundle:
    """Like :func:`make_training_bundle` but reusing pre-fitted z-stats.

    This avoids leaking target z-stats into the prior pool's training matrix.
    """
    needed = ["cot_linear"] if objective == "cot" else (
        ["sus", "nrs", "tlx"] if objective == "survey" else
        ["cot_linear", "sus", "nrs", "tlx"]
    )
    rows = df.dropna(subset=list(INPUT_FEATURES) + needed).copy()
    X = assemble_design_matrix(rows, feature_stats=feature_stats)
    y = compose_objective(
        rows, objective=objective, stats=output_stats, w_cot=w_cot, w_survey=w_survey
    ).to_numpy(dtype=float)
    return ObjectiveBundle(
        X_train=X, y_train=y,
        feature_stats=feature_stats, output_stats=output_stats,
        train_df=rows.reset_index(drop=True),
    )


def run_personalised_bo(
    target_mih: str,
    *,
    mode: PriorMode,
    df: pd.DataFrame,
    objective: ObjectiveKind = "combined",
    n_iter: int = 10,
    seed_n: int = 1,
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    kernel: KernelName = "matern52",
    acquisition: AcquisitionName = "EI",
    n_restarts: int = 3,
    refit_every: int = 1,
    seed: int | None = 0,
) -> BOResult:
    """High-level wrapper around :func:`simulate_bo`.

    Identical signature to ``simulate_bo`` -- exposed at module top to match
    the plan's public-API name.
    """
    return simulate_bo(
        target_mih,
        df=df, mode=mode, objective=objective,
        n_iter=n_iter, seed_n=seed_n,
        w_cot=w_cot, w_survey=w_survey,
        kernel=kernel, acquisition=acquisition,
        n_restarts=n_restarts, refit_every=refit_every, seed=seed,
    )
