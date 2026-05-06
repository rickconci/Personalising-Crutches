"""Personalised GP+BO API endpoints (new pipeline).

This module wires the hand-written GP + BO pipeline (``src/optimization``) to
the frontend's "Personalised BO" dashboard. It maintains an in-process
``BOSession`` registry keyed by UUID to cache the fitted GP across the user's
fit -> suggest -> submit loop, while persisting each new live observation to
the existing ``trials`` table with ``source='bo'``.

Data sources merged into the GP training pool, in order:

1. JSON prior pool from ``data_V2/`` (the grid-search dataset).
2. DB ``trials`` rows where ``source='bo'`` and not soft-deleted (i.e. all
   prior live BO observations).
3. The current session's in-memory observations (which also get written to
   the DB on submit).

All endpoints return JSON. Bodies are Pydantic models for cleanliness.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.connection import get_db
from database.models import CrutchGeometry as SQLCrutchGeometry
from database.models import Participant as SQLParticipant
from database.models import Trial as SQLTrial

from ..core.config import settings
from optimization.bo import (  # type: ignore[import-not-found]
    AcquisitionName,
    suggest_next as bo_suggest_next,
)
from optimization.data import (  # type: ignore[import-not-found]
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
    load_db_bo_trials,
    load_joined_dataset,
)
from optimization.gp import GPModel, KernelName, fit_gp_hyperparameters
from optimization.viz import (  # type: ignore[import-not-found]
    plot_correlation_matrix,
    plot_outcome_per_geometry,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
def _data_root() -> Path:
    """Return the ``data_V2`` directory next to ``settings.data_directory``.

    Falls back to the repo's ``data_V2`` if the configured directory does not
    exist (development convenience).
    """
    candidate = Path(settings.data_directory) if settings.data_directory else None
    if candidate and (candidate / "participants_gridsearch.json").exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[3]
    fallback = repo_root / "data_V2"
    if not fallback.exists():
        raise RuntimeError(
            f"Cannot locate data_V2 directory. Tried {candidate} and {fallback}"
        )
    return fallback


# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------
PriorModeStr = Literal["pool_with_self", "pool_loo", "no_prior"]


class StartSessionRequest(BaseModel):
    participant_mih: str = Field(..., description="MIH id matching Participant.name")
    mode: PriorModeStr = "pool_with_self"
    objective: ObjectiveKind = "combined"
    w_cot: float = 1.0
    w_survey: float = 1.0
    kernel: KernelName = "matern52"
    acquisition: AcquisitionName = "EI"


class SessionIdRequest(BaseModel):
    session_id: str


class SuggestRequest(BaseModel):
    session_id: str
    exclude_indices: list[int] | None = None
    top_k: int = 5


class SurveyResponsesIn(BaseModel):
    sus_q1: int
    sus_q2: int
    sus_q3: int
    sus_q4: int
    sus_q5: int
    sus_q6: int
    nrs_score: int
    tlx_mental_demand: int
    tlx_physical_demand: int
    tlx_performance: int
    tlx_effort: int
    tlx_frustration: int


class SubmitObservationRequest(BaseModel):
    session_id: str
    alpha: float
    beta: float
    gamma: float
    metabolic_power_W_per_kg: float
    walking_time_s: float
    distance_m: float
    survey: SurveyResponsesIn | None = None


# ---------------------------------------------------------------------------
# Survey scoring (mirrors src/frontend/js/systematic-mode/survey-manager.js)
# ---------------------------------------------------------------------------
_SUS_POSITIVE = {"sus_q1": True, "sus_q2": True, "sus_q3": False, "sus_q4": True,
                 "sus_q5": False, "sus_q6": False}
_TLX_FIELDS = ("tlx_mental_demand", "tlx_physical_demand", "tlx_performance",
               "tlx_effort", "tlx_frustration")


def _compute_sus_score(survey: SurveyResponsesIn) -> float:
    """Adapted SUS over 6 questions, normalised to 0-100 (higher is better)."""
    raw = 0
    for q, positive in _SUS_POSITIVE.items():
        v = int(getattr(survey, q))
        raw += (v - 1) if positive else (5 - v)
    return (raw / 24.0) * 100.0


def _compute_tlx_score(survey: SurveyResponsesIn) -> float:
    """NASA-TLX mean over 5 dimensions on a 0-20 scale (lower is better).

    Performance is reverse-coded so high performance = low workload.
    """
    total = 0
    for f in _TLX_FIELDS:
        v = int(getattr(survey, f))
        total += (20 - v) if f == "tlx_performance" else v
    return total / 5.0


# ---------------------------------------------------------------------------
# BOSession
# ---------------------------------------------------------------------------
@dataclass
class BOConfig:
    """Frozen configuration for a BO session.

    Attributes:
        mode: Prior pool selection strategy (string form -- see ``PriorMode``).
        objective: ``"cot"``, ``"survey"`` or ``"combined"``.
        w_cot, w_survey: Weights for the combined objective.
        kernel: GP kernel.
        acquisition: ``"EI"`` or ``"UCB"``.
    """

    mode: PriorModeStr
    objective: ObjectiveKind
    w_cot: float
    w_survey: float
    kernel: KernelName
    acquisition: AcquisitionName


@dataclass
class BOSession:
    """In-process state for one BO dashboard session."""

    id: str
    participant_mih: str
    db_participant_id: int | None
    config: BOConfig
    candidate_grid: pd.DataFrame
    pool_df: pd.DataFrame  # JSON + DB prior; refreshed on submit
    feature_stats: ZScoreStats | None = None
    output_stats: ZScoreStats | None = None
    bundle: ObjectiveBundle | None = None
    candidate_X: np.ndarray | None = None
    model: GPModel | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    excluded_idx: set[int] = field(default_factory=set)


_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, BOSession] = {}


def _get_session(session_id: str) -> BOSession:
    with _SESSION_LOCK:
        sess = _SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"BO session {session_id} not found")
    return sess


def _put_session(sess: BOSession) -> None:
    with _SESSION_LOCK:
        _SESSIONS[sess.id] = sess


# ---------------------------------------------------------------------------
# Pool building (mode-aware)
# ---------------------------------------------------------------------------
def _build_pool_for_mode(
    full_df: pd.DataFrame, *, target_mih: str, mode: PriorModeStr
) -> pd.DataFrame:
    """Return the GP-training pool corresponding to ``mode``.

    Mirrors ``optimization.pipeline._split_pool`` but operates on a single
    pre-merged dataframe so we can include DB-acquired BO trials.
    """
    if mode == "pool_with_self":
        return full_df.copy()
    if mode == "pool_loo":
        return full_df[full_df["participant"] != target_mih].copy()
    if mode == "no_prior":
        return full_df[full_df["participant"] == target_mih].copy()
    raise HTTPException(status_code=400, detail=f"Unknown mode {mode!r}")


def _refresh_session_pool(sess: BOSession, db: Session) -> None:
    """Re-load JSON + DB BO trials and rebuild the session's training pool."""
    df_json = load_joined_dataset(_data_root())
    df_db = load_db_bo_trials(db)
    full = pd.concat([df_json, df_db], ignore_index=True) if not df_db.empty else df_json
    sess.pool_df = _build_pool_for_mode(
        full, target_mih=sess.participant_mih, mode=sess.config.mode
    )


def _participant_features_for(sess: BOSession, db: Session) -> pd.Series:
    """Return the participant features Series used to broadcast over the grid."""
    rows = sess.pool_df[sess.pool_df["participant"] == sess.participant_mih]
    if not rows.empty:
        return rows[list(PARTICIPANT_FEATURES)].iloc[0]

    # Mode is no_prior + this participant has no JSON rows yet -- fall back to DB.
    p = (
        db.query(SQLParticipant)
        .filter(SQLParticipant.name == sess.participant_mih)
        .first()
    )
    if p is None or not isinstance(p.characteristics, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot find participant features for {sess.participant_mih}",
        )
    chars = p.characteristics
    return pd.Series(
        dict(
            height=float(chars["height"]),
            weight=float(chars["weight"]),
            forearm_length=float(chars["forearm_length"]),
            age=float(chars["age"]),
            activity_level=float(chars["activity_level"]),
            sex_male=1.0 if str(chars.get("sex", "")).lower().startswith("m") else 0.0,
            prev_crutch=1.0 if bool(chars.get("previous_crutch_experience")) else 0.0,
        )
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/participants")
async def list_bo_participants(db: Session = Depends(get_db)) -> dict[str, Any]:
    """List participants available for BO (intersection of JSON pool and DB)."""
    root = _data_root()
    raw = json.loads((root / "participants_gridsearch.json").read_text())
    db_by_name = {p.name: p.id for p in db.query(SQLParticipant).all()}

    out = []
    for mih, info in raw.items():
        out.append(
            dict(
                mih=mih,
                db_participant_id=db_by_name.get(mih),
                height=info.get("height"),
                weight=info.get("weight"),
                forearm_length=info.get("forearm_length"),
                age=info.get("age"),
                sex=info.get("sex"),
                activity_level=info.get("activity_level"),
                previous_crutch_experience=info.get("previous_crutch_experience"),
            )
        )
    out.sort(key=lambda r: r["mih"])
    return {"participants": out}


@router.get("/candidate-grid")
async def get_candidate_grid() -> dict[str, Any]:
    """Return the full discrete BO candidate grid (140 combos by default)."""
    grid = build_candidate_grid()
    return {
        "n_candidates": len(grid),
        "candidates": grid.to_dict(orient="records"),
    }


@router.post("/start-session")
async def start_session(
    body: StartSessionRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Create a new BO session and return its id + initial pool stats."""
    # Validate the participant exists in JSON pool (required for participant features).
    df_json = load_joined_dataset(_data_root())
    if body.participant_mih not in set(df_json["participant"]):
        raise HTTPException(
            status_code=404,
            detail=(
                f"participant_mih={body.participant_mih!r} not found in "
                "data_V2/participants_gridsearch.json"
            ),
        )
    db_participant = (
        db.query(SQLParticipant)
        .filter(SQLParticipant.name == body.participant_mih)
        .first()
    )

    cfg = BOConfig(
        mode=body.mode,
        objective=body.objective,
        w_cot=body.w_cot,
        w_survey=body.w_survey,
        kernel=body.kernel,
        acquisition=body.acquisition,
    )
    sess = BOSession(
        id=str(uuid.uuid4()),
        participant_mih=body.participant_mih,
        db_participant_id=db_participant.id if db_participant else None,
        config=cfg,
        candidate_grid=build_candidate_grid(),
        pool_df=pd.DataFrame(),  # filled below
    )
    _refresh_session_pool(sess, db)
    _put_session(sess)

    return {
        "session_id": sess.id,
        "participant_mih": sess.participant_mih,
        "db_participant_id": sess.db_participant_id,
        "config": cfg.__dict__,
        "n_pool_rows": int(len(sess.pool_df)),
        "n_candidates": int(len(sess.candidate_grid)),
        "pool_breakdown": sess.pool_df["participant"].value_counts().to_dict(),
    }


def _make_bundle_from_pool(sess: BOSession) -> ObjectiveBundle:
    """Re-fit z-stats on the current pool and assemble (X, y) for the GP."""
    cfg = sess.config
    needed = ["cot_linear"] if cfg.objective == "cot" else (
        ["sus", "nrs", "tlx"] if cfg.objective == "survey" else
        ["cot_linear", "sus", "nrs", "tlx"]
    )
    df = sess.pool_df.dropna(subset=list(INPUT_FEATURES) + needed).copy()
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                "GP training pool is empty for the selected (mode, objective). "
                "If mode=no_prior, submit at least one observation first."
            ),
        )
    feature_stats = fit_zscore(df, INPUT_FEATURES)
    output_stats = fit_zscore(df, ["cot_linear", "sus", "nrs", "tlx"])
    X = assemble_design_matrix(df, feature_stats=feature_stats)
    y = compose_objective(
        df, objective=cfg.objective, stats=output_stats,
        w_cot=cfg.w_cot, w_survey=cfg.w_survey,
    ).to_numpy(dtype=float)
    bundle = ObjectiveBundle(
        X_train=X, y_train=y,
        feature_stats=feature_stats, output_stats=output_stats,
        train_df=df.reset_index(drop=True),
    )
    sess.feature_stats = feature_stats
    sess.output_stats = output_stats
    return bundle


@router.post("/fit-gp")
async def fit_gp(body: SessionIdRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Fit GP hyperparameters on the current pool and cache it on the session."""
    sess = _get_session(body.session_id)
    _refresh_session_pool(sess, db)

    bundle = _make_bundle_from_pool(sess)
    model = fit_gp_hyperparameters(
        bundle.X_train, bundle.y_train,
        kernel=sess.config.kernel, n_restarts=3, seed=0,
    )

    target_features = _participant_features_for(sess, db)
    candidate_X = assemble_candidate_matrix(
        sess.candidate_grid, target_features, feature_stats=bundle.feature_stats,
    )

    sess.bundle = bundle
    sess.model = model
    sess.candidate_X = candidate_X
    _put_session(sess)

    signal_var, lengthscale, noise_var = (float(x) for x in model.theta)
    return {
        "session_id": sess.id,
        "kernel": model.kernel_name,
        "theta": dict(signal_variance=signal_var, lengthscale=lengthscale, noise_variance=noise_var),
        "train_nll": float(model.train_nll),
        "n_train": int(bundle.X_train.shape[0]),
        "n_candidates": int(candidate_X.shape[0]),
    }


@router.post("/suggest-next")
async def suggest_next_endpoint(body: SuggestRequest) -> dict[str, Any]:
    """Run the acquisition over the candidate grid and return the top suggestion."""
    sess = _get_session(body.session_id)
    if sess.model is None or sess.candidate_X is None or sess.bundle is None:
        raise HTTPException(status_code=400, detail="Call /fit-gp before /suggest-next")

    # Combine session-tracked excluded indices with the request's overrides.
    excluded = set(sess.excluded_idx)
    if body.exclude_indices:
        excluded.update(int(i) for i in body.exclude_indices)

    sug = bo_suggest_next(
        sess.model, sess.candidate_X, sess.bundle.y_train,
        acquisition=sess.config.acquisition,
        excluded_indices=excluded,
    )

    grid = sess.candidate_grid
    chosen = grid.iloc[sug.index]
    posterior_std = np.sqrt(np.clip(sug.var, 0.0, None))

    # Top-K alternatives (highest acquisition, also masking excluded)
    alpha_masked = sug.alpha.copy()
    if excluded:
        alpha_masked[np.array(sorted(excluded), dtype=int)] = -np.inf
    k = max(1, int(body.top_k))
    top_k_idx = np.argsort(-alpha_masked)[:k]
    top_k = []
    for idx in top_k_idx:
        idx = int(idx)
        if not np.isfinite(alpha_masked[idx]):
            continue
        row = grid.iloc[idx]
        top_k.append(
            dict(
                idx=idx,
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                gamma=float(row["gamma"]),
                acquisition_value=float(sug.alpha[idx]),
                posterior_mean=float(sug.mu[idx]),
                posterior_std=float(posterior_std[idx]),
            )
        )

    return {
        "session_id": sess.id,
        "suggested": dict(
            idx=int(sug.index),
            alpha=float(chosen["alpha"]),
            beta=float(chosen["beta"]),
            gamma=float(chosen["gamma"]),
            acquisition_value=float(sug.alpha[sug.index]),
            posterior_mean=float(sug.mu[sug.index]),
            posterior_std=float(posterior_std[sug.index]),
        ),
        "candidates": [
            dict(
                idx=i,
                alpha=float(grid.iloc[i]["alpha"]),
                beta=float(grid.iloc[i]["beta"]),
                gamma=float(grid.iloc[i]["gamma"]),
                posterior_mean=float(sug.mu[i]),
                posterior_std=float(posterior_std[i]),
                acquisition_value=float(sug.alpha[i]),
                excluded=bool(i in excluded),
            )
            for i in range(len(grid))
        ],
        "top_k_alternatives": top_k,
        "f_best": float(sess.bundle.y_train.min()) if len(sess.bundle.y_train) else None,
    }


def _find_or_create_geometry(
    db: Session, alpha: float, beta: float, gamma: float, delta: float = 0.0
) -> SQLCrutchGeometry:
    """Mirror the experiment_service find-or-create pattern (delta defaults to 0)."""
    geom = (
        db.query(SQLCrutchGeometry)
        .filter(
            SQLCrutchGeometry.alpha == alpha,
            SQLCrutchGeometry.beta == beta,
            SQLCrutchGeometry.gamma == gamma,
            SQLCrutchGeometry.delta == delta,
        )
        .first()
    )
    if geom:
        return geom
    name = f"BO_a{alpha:g}_b{beta:g}_g{gamma:g}"
    geom = SQLCrutchGeometry(
        name=name, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    )
    db.add(geom)
    db.flush()
    return geom


@router.post("/submit-observation")
async def submit_observation(
    body: SubmitObservationRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Persist a new BO trial and refresh the session's training pool."""
    sess = _get_session(body.session_id)
    if sess.db_participant_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Participant {sess.participant_mih} has no DB record; create "
                "them via the systematic mode first."
            ),
        )
    if body.walking_time_s <= 0 or body.distance_m <= 0:
        raise HTTPException(status_code=400, detail="walking_time_s and distance_m must be > 0")

    velocity = body.distance_m / body.walking_time_s
    cot = body.metabolic_power_W_per_kg / velocity

    survey = body.survey
    sus_score = _compute_sus_score(survey) if survey is not None else None
    tlx_score = _compute_tlx_score(survey) if survey is not None else None

    geom = _find_or_create_geometry(db, body.alpha, body.beta, body.gamma)

    processed_features = dict(
        bo_session_id=sess.id,
        metabolic_power_W_per_kg=body.metabolic_power_W_per_kg,
        walking_time_s=body.walking_time_s,
        distance_m=body.distance_m,
        velocity_m_s=velocity,
        cot_J_per_kg_m=cot,
        objective=sess.config.objective,
        w_cot=sess.config.w_cot,
        w_survey=sess.config.w_survey,
    )

    survey_kwargs: dict[str, Any] = {}
    if survey is not None:
        survey_kwargs.update(
            sus_q1=survey.sus_q1, sus_q2=survey.sus_q2, sus_q3=survey.sus_q3,
            sus_q4=survey.sus_q4, sus_q5=survey.sus_q5, sus_q6=survey.sus_q6,
            sus_score=sus_score,
            nrs_score=survey.nrs_score,
            tlx_mental_demand=survey.tlx_mental_demand,
            tlx_physical_demand=survey.tlx_physical_demand,
            tlx_performance=survey.tlx_performance,
            tlx_effort=survey.tlx_effort,
            tlx_frustration=survey.tlx_frustration,
            tlx_score=int(round(tlx_score)) if tlx_score is not None else None,
        )

    trial = SQLTrial(
        participant_id=sess.db_participant_id,
        geometry_id=geom.id,
        alpha=body.alpha,
        beta=body.beta,
        gamma=body.gamma,
        delta=0.0,
        source="bo",
        processed_features=processed_features,
        metabolic_cost=cot,
        **survey_kwargs,
    )
    db.add(trial)
    db.commit()
    db.refresh(trial)

    # Also mark this candidate-grid index as visited so the next /suggest-next
    # never proposes the same combination twice in this session.
    grid = sess.candidate_grid
    match = grid[
        (grid["alpha"] == body.alpha)
        & (grid["beta"] == body.beta)
        & (grid["gamma"] == body.gamma)
    ]
    if not match.empty:
        sess.excluded_idx.add(int(match.index[0]))

    sess.history.append(
        dict(
            iteration=len(sess.history) + 1,
            trial_id=int(trial.id),
            alpha=body.alpha, beta=body.beta, gamma=body.gamma,
            metabolic_power_W_per_kg=body.metabolic_power_W_per_kg,
            walking_time_s=body.walking_time_s,
            distance_m=body.distance_m,
            cot=cot,
            sus_score=sus_score, nrs_score=getattr(survey, "nrs_score", None),
            tlx_score=tlx_score,
        )
    )

    _refresh_session_pool(sess, db)
    _put_session(sess)

    best_y: float | None = None
    if sess.bundle is not None and len(sess.bundle.y_train):
        best_y = float(sess.bundle.y_train.min())

    return {
        "session_id": sess.id,
        "trial_id": int(trial.id),
        "cot": cot,
        "velocity_m_s": velocity,
        "sus_score": sus_score, "tlx_score": tlx_score,
        "n_pool_rows": int(len(sess.pool_df)),
        "n_history": len(sess.history),
        "history": sess.history,
        "best_y_before_refit": best_y,
        "needs_refit": True,
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Snapshot of the session for resume / debugging."""
    sess = _get_session(session_id)
    payload: dict[str, Any] = dict(
        session_id=sess.id,
        participant_mih=sess.participant_mih,
        db_participant_id=sess.db_participant_id,
        config=sess.config.__dict__,
        n_pool_rows=int(len(sess.pool_df)),
        n_candidates=int(len(sess.candidate_grid)),
        history=sess.history,
        excluded_idx=sorted(sess.excluded_idx),
        gp_fitted=sess.model is not None,
    )
    if sess.model is not None:
        sv, ls, nv = (float(x) for x in sess.model.theta)
        payload["theta"] = dict(signal_variance=sv, lengthscale=ls, noise_variance=nv)
        payload["train_nll"] = float(sess.model.train_nll)
        payload["n_train"] = (
            int(sess.bundle.X_train.shape[0]) if sess.bundle is not None else None
        )
    return payload


@router.get("/insights")
async def get_insights(
    objective: ObjectiveKind = "combined",
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return Plotly figure JSONs for cross-participant visualisations."""
    df_json = load_joined_dataset(_data_root())
    df_db = load_db_bo_trials(db)
    df = pd.concat([df_json, df_db], ignore_index=True) if not df_db.empty else df_json

    fig_outcome = plot_outcome_per_geometry(
        df, outcome="combined" if objective == "combined" else (
            "cot_linear" if objective == "cot" else "survey"
        ),
        w_cot=w_cot, w_survey=w_survey, title_prefix="all participants: ",
    )
    fig_corr = plot_correlation_matrix(df, method="pearson", title_prefix="all participants: ")

    return {
        "outcome_per_geometry": json.loads(fig_outcome.to_json()),
        "correlation_matrix": json.loads(fig_corr.to_json()),
        "n_rows": int(len(df)),
        "objective": objective,
    }
