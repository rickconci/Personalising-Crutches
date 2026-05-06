"""Cross-participant visualisations for the GP+BO pipeline (Plotly)."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import (
    INPUT_FEATURES,
    OUTPUT_COLUMNS,
    SURVEY_SIGNS,
)

OutcomeKind = Literal["cot_linear", "survey", "combined"]


# ---------------------------------------------------------------------------
# Outcome per geometry
# ---------------------------------------------------------------------------
def _geometry_label(row: pd.Series) -> str:
    return f"a{int(row['alpha'])}-b{int(row['beta'])}-g{int(row['gamma']):+d}"


def _outcome_series(df: pd.DataFrame, outcome: OutcomeKind, *,
                    w_cot: float = 1.0, w_survey: float = 1.0) -> pd.Series:
    """Compute the per-row outcome scalar for the chosen ``outcome`` (no z-scoring).

    Used by :func:`plot_outcome_per_geometry` to compute the **per-row,
    per-participant** value that gets ranked. Convention: **smaller = better**.

    * ``cot_linear``: raw CoT (J·kg⁻¹·m⁻¹), already lower-better.
    * ``survey``: raw signed sum ``−SUS + NRS + TLX``; lower-better.
    * ``combined``: ``w_cot * cot_linear + w_survey * (−SUS + NRS + TLX)`` in
      **raw units** (no z-scoring). The combined plot does **not** use this
      directly though — it builds the composite from per-participant **ranks**
      of CoT and survey instead, see :func:`_per_participant_rank_y`.
    """
    if outcome == "cot_linear":
        return df["cot_linear"]
    survey_raw = (
        SURVEY_SIGNS["sus"] * df["sus"]
        + SURVEY_SIGNS["nrs"] * df["nrs"]
        + SURVEY_SIGNS["tlx"] * df["tlx"]
    )
    if outcome == "survey":
        return survey_raw
    if outcome == "combined":
        return w_cot * df["cot_linear"] + w_survey * survey_raw
    raise ValueError(f"Unknown outcome={outcome!r}")


def _per_participant_rank_y(
    work: pd.DataFrame,
    outcome: OutcomeKind,
    *,
    w_cot: float = 1.0,
    w_survey: float = 1.0,
) -> pd.Series:
    """Per-row rank inside each participant; smaller rank = better.

    For ``cot_linear`` and ``survey`` we rank the corresponding raw column.
    For ``combined`` we rank ``cot_linear`` and ``survey`` **separately within
    each participant**, then take a weighted average of the two ranks. This
    avoids letting either column's scale dominate (which is exactly the
    z-score pitfall we hit before).
    """
    grp = work.groupby("participant", group_keys=False, sort=False)
    if outcome == "cot_linear":
        return grp["cot_linear"].rank(method="average", ascending=True)
    if outcome == "survey":
        survey_raw = (
            SURVEY_SIGNS["sus"] * work["sus"]
            + SURVEY_SIGNS["nrs"] * work["nrs"]
            + SURVEY_SIGNS["tlx"] * work["tlx"]
        )
        tmp = work.assign(__svy=survey_raw)
        return tmp.groupby("participant", group_keys=False, sort=False)["__svy"].rank(
            method="average", ascending=True
        )
    if outcome == "combined":
        survey_raw = (
            SURVEY_SIGNS["sus"] * work["sus"]
            + SURVEY_SIGNS["nrs"] * work["nrs"]
            + SURVEY_SIGNS["tlx"] * work["tlx"]
        )
        tmp = work.assign(__svy=survey_raw)
        rank_cot = tmp.groupby("participant", group_keys=False, sort=False)["cot_linear"].rank(
            method="average", ascending=True
        )
        rank_svy = tmp.groupby("participant", group_keys=False, sort=False)["__svy"].rank(
            method="average", ascending=True
        )
        denom = max(w_cot + w_survey, 1e-12)
        return (w_cot * rank_cot + w_survey * rank_svy) / denom
    raise ValueError(f"Unknown outcome={outcome!r}")


def plot_outcome_per_geometry(
    df: pd.DataFrame,
    *,
    outcome: OutcomeKind = "cot_linear",
    w_cot: float = 1.0,
    w_survey: float = 1.0,
    title_prefix: str = "",
    show_individual_points: bool = True,
) -> go.Figure:
    """Bar chart of per-geometry outcome (mean ± std) across participants.

    Geometries on the horizontal axis are **ordered by mean within-participant
    rank**, not by the raw mean outcome. Inside each participant, rows are ranked
    on ``__y`` with ``ascending=True`` (rank ``1`` = best locally for ``cot``,
    ``survey`` and ``combined`` as defined by :func:`_outcome_series`). That makes
    the combined plot reflect “how often this geometry is near-best” instead of
    letting z-score magnitudes distort the ordering.

    Args:
        df: Joined DataFrame from :func:`data.load_joined_dataset`.
        outcome: ``"cot_linear"`` (raw CoT — lower ``__y`` better),
            ``"survey"`` (raw ``−SUS + NRS + TLX`` — lower ``__y`` better),
            or ``"combined"`` (z-scored weighted composite — lower ``__y`` better).
        w_cot, w_survey: Weights used when ``outcome="combined"``.
        title_prefix: Text prepended to the title.
        show_individual_points: Overlay one scatter point per participant per
            geometry to visualise the spread.

    Returns:
        Plotly figure.
    """
    work = df.copy()
    needed = {"cot_linear": ["cot_linear"], "survey": ["sus", "nrs", "tlx"],
              "combined": ["cot_linear", "sus", "nrs", "tlx"]}[outcome]
    work = work.dropna(subset=needed + ["participant"]).copy()
    work["__geom"] = work.apply(_geometry_label, axis=1)
    work["__rank"] = _per_participant_rank_y(work, outcome, w_cot=w_cot, w_survey=w_survey)
    work["__raw"] = _outcome_series(work, outcome, w_cot=w_cot, w_survey=w_survey)

    summary = (
        work.groupby("__geom")
        .agg(
            mean_rank=("__rank", "mean"),
            std_rank=("__rank", "std"),
            mean_raw=("__raw", "mean"),
            std_raw=("__raw", "std"),
            count=("__rank", "count"),
        )
        .reset_index()
        .sort_values("mean_rank", ascending=True)
    )
    geom_order = summary["__geom"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["__geom"],
            y=summary["mean_rank"],
            error_y=dict(type="data", array=summary["std_rank"].fillna(0.0), visible=True),
            name="mean rank ± std",
            marker=dict(color="#5B9BD5"),
            hovertemplate=(
                "geometry=%{x}<br>"
                "mean rank=%{y:.2f}<br>"
                "std rank=%{customdata[0]:.2f}<br>"
                "mean raw=%{customdata[1]:.3f}<br>"
                "n=%{customdata[2]}<extra></extra>"
            ),
            customdata=np.stack(
                [
                    summary["std_rank"].fillna(0.0).values,
                    summary["mean_raw"].fillna(np.nan).values,
                    summary["count"].astype(int).values,
                ],
                axis=-1,
            ),
        )
    )

    if show_individual_points:
        work_plot = work.copy()
        geom_cat = pd.Categorical(work_plot["__geom"], categories=geom_order, ordered=True)
        work_plot = work_plot.assign(__geom_ordered=geom_cat).sort_values(
            "__geom_ordered", kind="stable"
        )
        fig.add_trace(
            go.Scatter(
                x=work_plot["__geom"],
                y=work_plot["__rank"],
                mode="markers",
                name="participants (rank)",
                marker=dict(color="rgba(0,0,0,0.4)", size=6, symbol="circle"),
                hovertext=[
                    f"{p} {tk} · raw={r:.3f}"
                    for p, tk, r in zip(
                        work_plot["participant"], work_plot["trial_key"], work_plot["__raw"]
                    )
                ],
                hovertemplate="%{hovertext}<br>%{x}: rank %{y:.2f}<extra></extra>",
            )
        )

    n_geoms_per_ppt = work.groupby("participant")["__geom"].nunique().median()
    fig.update_layout(
        title=(
            f"{title_prefix}outcome per geometry — {outcome} (rank-based)<br>"
            f'<sup style="font-size:11px;color:#444">'
            f"y = mean within-participant rank · 1 = best of that person's geometries · "
            f"~{n_geoms_per_ppt:.0f} ranks per ppt</sup>"
        ),
        xaxis_title="geometry (alpha-beta-gamma)",
        yaxis_title="mean within-participant rank (1 = best)",
        bargap=0.2,
        showlegend=show_individual_points,
        width=950, height=520,
    )
    fig.update_yaxes(autorange="reversed")  # rank 1 sits at the top
    fig.update_xaxes(categoryarray=geom_order, categoryorder="array")
    return fig


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------
def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    inputs: Iterable[str] = INPUT_FEATURES,
    outputs: Iterable[str] = OUTPUT_COLUMNS,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    title_prefix: str = "",
) -> go.Figure:
    """Correlation heatmap between input features and output objectives.

    The matrix is rectangular (rows = ``inputs``, cols = ``outputs``) so it is
    easy to read which body/geometry features drive which outcome.

    Args:
        df: Joined dataset.
        inputs, outputs: Column subsets to correlate.
        method: ``"pearson"`` (default), ``"spearman"`` or ``"kendall"``.
        title_prefix: Text prepended to the title.

    Returns:
        Plotly heatmap figure.
    """
    inputs = list(inputs)
    outputs = list(outputs)
    sub = df[inputs + outputs].astype(float)

    # Pairwise corr handles NaNs row-wise; pandas .corr already does this with
    # min_periods. We compute the full corr then slice to the rectangular block.
    full = sub.corr(method=method)
    mat = full.loc[inputs, outputs].to_numpy()

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=outputs,
            y=inputs,
            colorscale="RdBu",
            zmin=-1.0, zmax=1.0,
            colorbar=dict(title=f"{method} ρ"),
            hovertemplate="input=%{y}<br>output=%{x}<br>ρ=%{z:.3f}<extra></extra>",
        )
    )
    annotations = []
    for i, row in enumerate(inputs):
        for j, col in enumerate(outputs):
            v = mat[i, j]
            annotations.append(
                dict(
                    x=col, y=row, text=f"{v:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(v) > 0.5 else "black", size=11),
                )
            )
    fig.update_layout(
        title=f"{title_prefix}input × output correlations ({method})",
        annotations=annotations,
        xaxis_title="outputs",
        yaxis_title="inputs",
        width=720, height=540,
    )
    fig.update_yaxes(autorange="reversed")
    return fig
