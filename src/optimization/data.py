"""Data loading, joining and objective construction for GP+BO personalisation.

The pipeline expects the three JSON sources produced upstream:

* ``data_V2/participants_gridsearch.json`` -- per-MIH covariates.
* ``data_V2/metabolics_results.json``      -- per-MIH per-trial CoT results.
* ``data_V2/trials_gridsearch.json``       -- per-trial questionnaire results.

This module joins them into a single long-format ``pd.DataFrame`` (one row per
metabolics trial), and provides helpers to build the dense BO candidate grid,
fit/apply z-scores, and assemble the GP design matrix + minimisation target.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid heavy ORM import at module load time
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARTICIPANT_FEATURES: tuple[str, ...] = (
    "height",
    "weight",
    "forearm_length",
    "age",
    "activity_level",
    "sex_male",
    "prev_crutch",
)

GEOMETRY_FEATURES: tuple[str, ...] = ("alpha", "beta", "gamma")

INPUT_FEATURES: tuple[str, ...] = PARTICIPANT_FEATURES + GEOMETRY_FEATURES

OUTPUT_COLUMNS: tuple[str, ...] = ("cot_linear", "sus", "nrs", "tlx")

#: Survey-composite uses -SUS + NRS + TLX (z-scored), so smaller is better.
SURVEY_SIGNS: dict[str, float] = {"sus": -1.0, "nrs": +1.0, "tlx": +1.0}

#: Default denser candidate grid steps (per design choice).
DEFAULT_ALPHA_STEP: float = 7.5
DEFAULT_BETA_STEP: float = 7.5
DEFAULT_GAMMA_STEP: float = 3.0

DEFAULT_ALPHA_RANGE: tuple[float, float] = (85.0, 105.0)
DEFAULT_BETA_RANGE: tuple[float, float] = (95.0, 125.0)
DEFAULT_GAMMA_RANGE: tuple[float, float] = (-9.0, 9.0)

ObjectiveKind = Literal["cot", "survey", "combined"]


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------
_GEOMETRY_RE = re.compile(r"^G(m|p)?(\d+)_A(\d+)_B(\d+)$")


def parse_geometry_string(geom: str) -> tuple[float, float, float]:
    """Parse a ``"G0_A95_B95"`` style string into ``(alpha, beta, gamma)``.

    Args:
        geom: e.g. ``"G0_A95_B95"``, ``"Gm9_A85_B125"``, ``"Gp9_A105_B95"``.

    Returns:
        ``(alpha, beta, gamma)`` as floats. Note the order matches the GP
        feature ordering, while the gamma sign comes from the ``Gm``/``Gp``
        prefix.

    Raises:
        ValueError: If ``geom`` does not match the expected pattern.
    """
    m = _GEOMETRY_RE.match(geom)
    if m is None:
        raise ValueError(f"Cannot parse geometry string: {geom!r}")
    sign_tok, gamma_mag, alpha_s, beta_s = m.groups()
    gamma = float(gamma_mag)
    if sign_tok == "m":
        gamma = -gamma
    return float(alpha_s), float(beta_s), gamma


# ---------------------------------------------------------------------------
# Joined dataset loader
# ---------------------------------------------------------------------------
def _load_participants(participants_json: Path) -> pd.DataFrame:
    """Load participant covariates as a tidy DataFrame keyed by ``participant``."""
    raw = json.loads(participants_json.read_text())
    rows = []
    for mih, p in raw.items():
        rows.append(
            dict(
                participant=mih,
                height=float(p["height"]),
                weight=float(p["weight"]),
                forearm_length=float(p["forearm_length"]),
                age=float(p["age"]),
                activity_level=float(p["activity_level"]),
                sex_male=1.0 if str(p["sex"]).lower().startswith("m") else 0.0,
                prev_crutch=1.0 if bool(p["previous_crutch_experience"]) else 0.0,
            )
        )
    return pd.DataFrame(rows)


def _load_metabolics_long(metabolics_json: Path) -> pd.DataFrame:
    """Flatten ``metabolics_results.json`` to one row per measured trial.

    Columns: ``participant, trial_key, alpha, beta, gamma, cot_linear, cot_raw``.
    Only trials with a parsable geometry are emitted; ``baseline_1``/
    ``baseline_2`` are kept (they share geometry ``G0_A95_B95``).
    """
    raw = json.loads(metabolics_json.read_text())
    rows = []
    for mih, mdata in raw.items():
        for trial_key, trial in mdata.items():
            if not isinstance(trial, dict) or "geometry" not in trial:
                continue
            try:
                alpha, beta, gamma = parse_geometry_string(trial["geometry"])
            except ValueError:
                continue
            cot_lin = trial.get("cot_linear_baseline_adj")
            cot_raw = trial.get("cot_raw")
            rows.append(
                dict(
                    participant=mih,
                    trial_key=trial_key,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    cot_linear=float(cot_lin) if cot_lin is not None else np.nan,
                    cot_raw=float(cot_raw) if cot_raw is not None else np.nan,
                )
            )
    return pd.DataFrame(rows)


def _load_trials_grouped(trials_json: Path) -> pd.DataFrame:
    """Aggregate ``trials_gridsearch.json`` into one row per geometry & participant.

    Multiple repeats at the same ``(participant, alpha, beta, gamma)`` are
    averaged (some participants like ``MIH15`` have repeated questionnaire
    sessions).
    """
    raw = json.loads(trials_json.read_text())
    df = pd.DataFrame(raw["trials"])
    df = df.rename(columns={"participant_name": "participant"})
    keep = ["participant", "alpha", "beta", "gamma", "sus_score", "nrs_score", "tlx_score"]
    df = df[keep].dropna(subset=["alpha", "beta", "gamma"])
    grouped = (
        df.groupby(["participant", "alpha", "beta", "gamma"], as_index=False)
        .mean(numeric_only=True)
        .rename(columns={"sus_score": "sus", "nrs_score": "nrs", "tlx_score": "tlx"})
    )
    return grouped


def load_joined_dataset(data_root: str | Path) -> pd.DataFrame:
    """Join participants, metabolics and questionnaire data into one DataFrame.

    The resulting frame has one row per metabolics trial (so two ``baseline_*``
    rows per participant), with input features and all four output columns
    (``cot_linear, sus, nrs, tlx``). Survey columns are aggregated per
    geometry per participant before the join, so ``baseline_1``/``baseline_2``
    inherit the same survey scores when applicable.

    Args:
        data_root: Path to the ``data_V2`` directory.

    Returns:
        A ``pd.DataFrame`` with columns
        ``[participant, trial_key, alpha, beta, gamma, *PARTICIPANT_FEATURES,
          cot_linear, cot_raw, sus, nrs, tlx]``.

    Example:
        >>> df = load_joined_dataset("data_V2")
        >>> df.shape[0] >= 100
        True
    """
    root = Path(data_root)
    participants = _load_participants(root / "participants_gridsearch.json")
    metabolics = _load_metabolics_long(root / "metabolics_results.json")
    trials = _load_trials_grouped(root / "trials_gridsearch.json")

    df = metabolics.merge(
        trials, on=["participant", "alpha", "beta", "gamma"], how="left"
    ).merge(participants, on="participant", how="left")

    cols = (
        ["participant", "trial_key", *GEOMETRY_FEATURES]
        + list(PARTICIPANT_FEATURES)
        + ["cot_linear", "cot_raw", "sus", "nrs", "tlx"]
    )
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------
def _arange_inclusive(lo: float, hi: float, step: float) -> np.ndarray:
    """``np.arange`` that always includes the endpoint ``hi``.

    >>> _arange_inclusive(85.0, 105.0, 7.5).tolist()
    [85.0, 92.5, 100.0, 105.0]
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    vals = list(np.arange(lo, hi + 1e-9, step))
    if not np.isclose(vals[-1], hi):
        vals.append(float(hi))
    return np.asarray(vals, dtype=float)


def build_candidate_grid(
    *,
    alpha_range: tuple[float, float] = DEFAULT_ALPHA_RANGE,
    beta_range: tuple[float, float] = DEFAULT_BETA_RANGE,
    gamma_range: tuple[float, float] = DEFAULT_GAMMA_RANGE,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    beta_step: float = DEFAULT_BETA_STEP,
    gamma_step: float = DEFAULT_GAMMA_STEP,
) -> pd.DataFrame:
    """Build the dense BO candidate grid over (alpha, beta, gamma).

    Defaults yield :math:`4 \\times 5 \\times 7 = 140` combinations spanning
    the buildable ranges.

    Returns:
        A DataFrame with columns ``[alpha, beta, gamma]``.

    Example:
        >>> g = build_candidate_grid()
        >>> g.shape
        (140, 3)
    """
    alphas = _arange_inclusive(*alpha_range, step=alpha_step)
    betas = _arange_inclusive(*beta_range, step=beta_step)
    gammas = _arange_inclusive(*gamma_range, step=gamma_step)

    a, b, g = np.meshgrid(alphas, betas, gammas, indexing="ij")
    grid = pd.DataFrame(
        dict(alpha=a.ravel(), beta=b.ravel(), gamma=g.ravel())
    )
    return grid


# ---------------------------------------------------------------------------
# Z-scoring helpers
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ZScoreStats:
    """Per-column mean/std fit on a training subset.

    Attributes:
        means: Column -> mean (fit on training rows).
        stds: Column -> std (clipped at ``eps`` to avoid divide-by-zero).
        eps: Floor applied to std before division.
    """

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    eps: float = 1e-9

    def transform(self, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        """Apply z-score to ``cols`` of ``df`` (out-of-place)."""
        out = df.copy()
        for c in cols:
            mu = self.means.get(c, 0.0)
            sd = max(self.stds.get(c, 1.0), self.eps)
            out[c] = (out[c] - mu) / sd
        return out


def fit_zscore(df: pd.DataFrame, cols: Iterable[str]) -> ZScoreStats:
    """Fit per-column z-stats on ``df``. NaNs are ignored.

    Example:
        >>> import pandas as pd
        >>> stats = fit_zscore(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), ["a"])
        >>> round(stats.means["a"], 3), round(stats.stds["a"], 3)
        (2.0, 0.816)
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for c in cols:
        col = df[c].astype(float)
        means[c] = float(np.nanmean(col))
        stds[c] = float(np.nanstd(col))
    return ZScoreStats(means=means, stds=stds)


# ---------------------------------------------------------------------------
# Objective + design matrix
# ---------------------------------------------------------------------------
def compose_survey_score(
    df: pd.DataFrame, stats: ZScoreStats
) -> pd.Series:
    """z(NRS) + z(TLX) - z(SUS), aligned so 'lower is better'."""
    z = stats.transform(df, ["sus", "nrs", "tlx"])
    return SURVEY_SIGNS["nrs"] * z["nrs"] + SURVEY_SIGNS["tlx"] * z["tlx"] + SURVEY_SIGNS["sus"] * z["sus"]


def compose_objective(
    df: pd.DataFrame,
    *,
    objective: ObjectiveKind,
    stats: ZScoreStats,
    w_cot: float = 1.0,
    w_survey: float = 1.0,
) -> pd.Series:
    """Build the scalar minimisation target ``y``.

    * ``"cot"``: ``z(cot_linear)``.
    * ``"survey"``: ``z(NRS) + z(TLX) - z(SUS)``.
    * ``"combined"``: ``w_cot * z(cot_linear) + w_survey * (z(NRS) + z(TLX) - z(SUS))``.

    NaNs propagate -- callers should drop them before passing rows to the GP.
    """
    z = stats.transform(df, ["cot_linear"])
    cot_term = z["cot_linear"]
    survey_term = compose_survey_score(df, stats)

    if objective == "cot":
        return cot_term
    if objective == "survey":
        return survey_term
    if objective == "combined":
        return w_cot * cot_term + w_survey * survey_term
    raise ValueError(f"Unknown objective={objective!r}")


def assemble_design_matrix(
    df: pd.DataFrame,
    *,
    feature_stats: ZScoreStats,
    feature_cols: Iterable[str] = INPUT_FEATURES,
) -> np.ndarray:
    """Return the z-scored feature matrix ``X`` ordered as ``feature_cols``."""
    z = feature_stats.transform(df, feature_cols)
    return z[list(feature_cols)].to_numpy(dtype=float)


@dataclass(slots=True)
class ObjectiveBundle:
    """Bundle of training matrices and the z-stats used to build them.

    Attributes:
        X_train: Z-scored design matrix of training rows.
        y_train: Z-scored minimisation target (vector).
        feature_stats: Z-stats fit on the training inputs.
        output_stats: Z-stats fit on the training outputs (cot + survey).
        train_df: The training rows after NaN filtering (for inspection).
    """

    X_train: np.ndarray
    y_train: np.ndarray
    feature_stats: ZScoreStats
    output_stats: ZScoreStats
    train_df: pd.DataFrame


def make_training_bundle(
    train_df: pd.DataFrame,
    *,
    objective: ObjectiveKind,
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    feature_cols: Iterable[str] = INPUT_FEATURES,
) -> ObjectiveBundle:
    """Drop NaNs for the chosen objective, fit z-stats, and assemble (X, y).

    Args:
        train_df: Pool of training rows.
        objective: ``"cot"``, ``"survey"`` or ``"combined"``.
        w_cot: Weight on the CoT term in ``"combined"`` mode.
        w_survey: Weight on the survey-composite term in ``"combined"`` mode.
        feature_cols: Input columns (defaults to participant + geometry).

    Returns:
        :class:`ObjectiveBundle`.
    """
    needed = ["cot_linear"] if objective == "cot" else (
        ["sus", "nrs", "tlx"] if objective == "survey" else
        ["cot_linear", "sus", "nrs", "tlx"]
    )
    df = train_df.dropna(subset=list(feature_cols) + needed).copy()
    feature_stats = fit_zscore(df, feature_cols)
    output_stats = fit_zscore(df, ["cot_linear", "sus", "nrs", "tlx"])

    X = assemble_design_matrix(df, feature_stats=feature_stats, feature_cols=feature_cols)
    y = compose_objective(
        df, objective=objective, stats=output_stats, w_cot=w_cot, w_survey=w_survey
    ).to_numpy(dtype=float)
    return ObjectiveBundle(X_train=X, y_train=y, feature_stats=feature_stats,
                            output_stats=output_stats, train_df=df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# DB extension: merge JSON prior pool with new BO-acquired trials in the DB
# ---------------------------------------------------------------------------
def _participant_characteristics_to_features(
    name: str, characteristics: dict[str, Any] | None
) -> dict[str, float] | None:
    """Convert a DB ``Participant.characteristics`` JSON blob to feature dict.

    Mirrors the JSON participant loader so DB-stored participants are featurised
    identically. Missing required fields cause this row to be dropped (returns
    ``None``) so the GP never sees half-NaN inputs.
    """
    if not isinstance(characteristics, dict):
        logger.debug("Participant %s has no characteristics; skipping", name)
        return None
    try:
        feats = dict(
            height=float(characteristics["height"]),
            weight=float(characteristics["weight"]),
            forearm_length=float(characteristics["forearm_length"]),
            age=float(characteristics["age"]),
            activity_level=float(characteristics["activity_level"]),
            sex_male=1.0 if str(characteristics.get("sex", "")).lower().startswith("m") else 0.0,
            prev_crutch=1.0 if bool(characteristics.get("previous_crutch_experience")) else 0.0,
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.debug("Participant %s missing required characteristic (%s); skipping", name, e)
        return None
    return feats


def load_db_bo_trials(db_session: "Session") -> pd.DataFrame:
    """Load DB rows where ``source == 'bo'`` and shape them like the joined dataset.

    Returns columns matching :func:`load_joined_dataset` so the two can be
    concatenated. Trials with insufficient data (no participant features, or
    missing ``cot_linear``) are dropped silently after a debug log.

    The CoT value is read from the ``metabolic_cost`` column (which the BO
    submit endpoint sets to ``power / velocity``); raw inputs sit in
    ``processed_features`` JSON for round-trip inspection.

    Args:
        db_session: SQLAlchemy session.

    Returns:
        ``pd.DataFrame`` (possibly empty) with the same columns as
        :func:`load_joined_dataset`.
    """
    # Local import to avoid pulling SQLAlchemy when the module is used in
    # notebooks without a DB session.
    from database.models import Trial  # type: ignore[import-not-found]

    rows = (
        db_session.query(Trial)
        .filter(Trial.source == "bo", Trial.deleted_at.is_(None))
        .all()
    )
    if not rows:
        return pd.DataFrame(
            columns=(
                ["participant", "trial_key", *GEOMETRY_FEATURES]
                + list(PARTICIPANT_FEATURES)
                + ["cot_linear", "cot_raw", "sus", "nrs", "tlx"]
            )
        )

    out: list[dict[str, Any]] = []
    for tr in rows:
        if tr.participant is None:
            continue
        feats = _participant_characteristics_to_features(
            tr.participant.name, tr.participant.characteristics
        )
        if feats is None:
            continue
        if tr.metabolic_cost is None:
            continue  # need CoT
        # Geometry: trial denormalised columns first, fall back to relationship
        alpha = tr.alpha if tr.alpha is not None else (tr.geometry.alpha if tr.geometry else None)
        beta = tr.beta if tr.beta is not None else (tr.geometry.beta if tr.geometry else None)
        gamma = tr.gamma if tr.gamma is not None else (tr.geometry.gamma if tr.geometry else None)
        if alpha is None or beta is None or gamma is None:
            continue

        out.append(
            dict(
                participant=tr.participant.name,
                trial_key=f"bo_{tr.id}",
                alpha=float(alpha),
                beta=float(beta),
                gamma=float(gamma),
                cot_linear=float(tr.metabolic_cost),
                cot_raw=float(tr.metabolic_cost),  # no baseline correction available
                sus=float(tr.sus_score) if tr.sus_score is not None else np.nan,
                nrs=float(tr.nrs_score) if tr.nrs_score is not None else np.nan,
                tlx=float(tr.tlx_score) if tr.tlx_score is not None else np.nan,
                **feats,
            )
        )

    if not out:
        return pd.DataFrame(
            columns=(
                ["participant", "trial_key", *GEOMETRY_FEATURES]
                + list(PARTICIPANT_FEATURES)
                + ["cot_linear", "cot_raw", "sus", "nrs", "tlx"]
            )
        )
    cols = (
        ["participant", "trial_key", *GEOMETRY_FEATURES]
        + list(PARTICIPANT_FEATURES)
        + ["cot_linear", "cot_raw", "sus", "nrs", "tlx"]
    )
    return pd.DataFrame(out)[cols]


def load_joined_dataset_with_db_bo(
    data_root: str | Path, db_session: "Session"
) -> pd.DataFrame:
    """Convenience: JSON prior pool + DB-acquired BO trials, concatenated.

    Args:
        data_root: ``data_V2`` directory.
        db_session: SQLAlchemy session for the experiments DB.

    Returns:
        Same schema as :func:`load_joined_dataset`, with extra rows for any
        ``source='bo'`` trials currently in the DB.
    """
    df_json = load_joined_dataset(data_root)
    df_db = load_db_bo_trials(db_session)
    if df_db.empty:
        return df_json
    return pd.concat([df_json, df_db], ignore_index=True)


def assemble_candidate_matrix(
    candidate_grid: pd.DataFrame,
    participant_features: pd.Series | dict[str, float],
    *,
    feature_stats: ZScoreStats,
    feature_cols: Iterable[str] = INPUT_FEATURES,
) -> np.ndarray:
    """Broadcast one participant's covariates over the candidate grid and z-score.

    Args:
        candidate_grid: DataFrame with at least the geometry columns.
        participant_features: Mapping of participant features for the target.
        feature_stats: Z-stats used to transform the inputs (must match
            training).
        feature_cols: Input columns (defaults to participant + geometry).

    Returns:
        ``(n_candidates, n_features)`` z-scored design matrix.
    """
    feat = dict(participant_features) if isinstance(participant_features, dict) else participant_features.to_dict()
    df = candidate_grid.copy()
    for col in PARTICIPANT_FEATURES:
        if col not in feat:
            raise KeyError(f"Missing participant feature {col!r}")
        df[col] = float(feat[col])
    return assemble_design_matrix(df, feature_stats=feature_stats, feature_cols=feature_cols)
