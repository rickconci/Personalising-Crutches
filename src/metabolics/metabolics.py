"""Metabolics pipeline for Cost-of-Transport extraction.

Pipeline per MIH session::

    parse_metabolics_csv -> compute_metabolic_power -> extract_markers
        -> select_trial_windows (analyst-reviewable) -> estimate_steady_state
        -> compute_cot -> apply_baseline_corrections -> save_results

The output ``data_V2/metabolics_results.json`` mirrors the trial-key structure
of ``data_V2/experiment_order.json`` so the Bayesian-optimisation loop can
read the cost-of-transport prior directly via ``(MIH, trial_key)``.

Each function is small, fully typed, and idempotent so it can be invoked
cell-by-cell from ``src/metabolics/metabolics.ipynb`` for visual validation.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Brockway (1987) constants converting (mL/min) of O2 / CO2 into Watts.
#   O2:  16.58 kJ/L * 1000 / 60s = 276.3 W per L/min  =  0.2763 W per (mL/min)
#   CO2: 4.51  kJ/L * 1000 / 60s =  75.2 W per L/min  =  0.0752 W per (mL/min)
BROCKWAY_VO2_W_PER_ML_MIN: float = 0.278
BROCKWAY_VCO2_W_PER_ML_MIN: float = 0.075

DEFAULT_TAU_S: float = 42.0
DEFAULT_MAX_WINDOW_S: float = 125.0  # 2:00 cap from start marker (protocol walking time)
DEFAULT_MARKER_DEDUP_S: float = 10.0  # collapse double-presses

#: Canonical chronological order of trial keys in ``experiment_order.json``.
TRIAL_KEYS: tuple[str, ...] = (
    "baseline_1",
    "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "11", "12",
    "baseline_2",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Marker:
    """A deduplicated marker placed by the analyst during the metabolics test."""

    index: int
    time_s: float
    item: int | None = None


@dataclass(slots=True)
class TrialWindow:
    """A time window (start..end) of one experimental trial within a session.

    ``start_marker_idx == -1`` indicates a synthetic start (the analyst chose
    an absolute time via ``start_s`` rather than picking an existing marker).
    """

    trial_key: str
    start_marker_idx: int  # -1 means "no marker; use start_s directly"
    start_s: float
    end_s: float
    duration_s: float
    notes: str | None = None


@dataclass(slots=True)
class SSResult:
    """Steady-state estimate from a single trial window."""

    y_ss: float
    y_ss_std: float
    tau: float
    tau_std: float
    r_squared: float
    n_points: int
    fit_method: str  # "exponential_rise" | "last30pct_avg(<reason>)" | "empty"
    residual_std: float


@dataclass(slots=True)
class CoTResult:
    """Cost-of-transport estimate (J / (kg * m)) for a single trial."""

    trial_key: str
    geometry: str | None
    marker_time_s: float
    window_end_s: float
    walking_time_s: float
    distance_m: float
    velocity_m_s: float
    steady_state_W_per_kg: float
    steady_state_std: float
    exp_fit_tau_s: float
    exp_fit_r2: float
    cot_raw: float
    cot_raw_std: float
    cot_linear_baseline_adj: float | None
    cot_ratio_baseline_norm: float | None
    fit_method: str
    notes: str | None = None


# ---------------------------------------------------------------------------
# File discovery & CSV parsing
# ---------------------------------------------------------------------------
def _mih_id_aliases(mih_id: str) -> list[str]:
    """Return on-disk directory name candidates for a JSON MIH id.

    The JSON uses zero-padded ids (``MIH02``) but a couple of session
    directories use the unpadded form (``MIH2``). This helper yields both
    spellings so the loader works regardless of which the user typed.
    """
    out = [mih_id]
    m = re.fullmatch(r"MIH0*(\d+)", mih_id)
    if m:
        n = int(m.group(1))
        out.extend([f"MIH{n}", f"MIH{n:02d}"])
    seen: set[str] = set()
    return [a for a in out if not (a in seen or seen.add(a))]


def find_metabolics_csv(data_root: Path, mih_id: str) -> Path:
    """Locate the single metabolics CSV for ``mih_id`` under ``data_root/raw``.

    Args:
        data_root: Path to ``data_V2`` (the directory that contains ``raw/``).
        mih_id: Subject identifier, e.g. ``"MIH01"``. Also accepts the
            unpadded variant (``"MIH2"`` ↔ ``"MIH02"``).

    Returns:
        Path to the metabolics CSV.

    Raises:
        FileNotFoundError: If the MIH directory, session dir or CSV is missing.
        RuntimeError: If multiple CSVs are found (ambiguous).
    """
    raw_root = Path(data_root) / "raw"
    base: Path | None = next(
        (raw_root / alias for alias in _mih_id_aliases(mih_id) if (raw_root / alias).is_dir()),
        None,
    )
    if base is None:
        raise FileNotFoundError(
            f"No MIH directory under {raw_root} for any of {_mih_id_aliases(mih_id)}"
        )
    sessions = sorted(p for p in base.iterdir() if p.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No session dirs under {base}")
    if len(sessions) > 1:
        logger.warning(
            "Multiple session dirs in %s (%s); picking first",
            base, [s.name for s in sessions],
        )
    session = sessions[0]
    metabolics_dir: Path | None = next(
        (session / d for d in ("Metabolics", "metabolics") if (session / d).is_dir()),
        None,
    )
    if metabolics_dir is None:
        raise FileNotFoundError(f"No Metabolics/metabolics dir under {session}")
    csvs = sorted(metabolics_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {metabolics_dir}")
    if len(csvs) > 1:
        raise RuntimeError(
            f"Multiple CSVs in {metabolics_dir}: {[c.name for c in csvs]}"
        )
    return csvs[0]


def parse_metabolics_csv(csv_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse a COSMED-style metabolics CSV into a tidy DataFrame and metadata dict.

    The file has a free-form header, a key-value metadata block (comma-separated),
    a ``IDS,...`` column-header row, and then breath-by-breath ``BxB`` rows.

    Args:
        csv_path: Path to the CSV.

    Returns:
        Tuple ``(df, meta)`` where ``df`` contains only ``BxB`` rows with a
        ``time_seconds`` column added, and ``meta`` is the parsed metadata
        (``Weight``, ``Height``, ``Firstname``, ...).
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(("IDS,", "IDs,"))),
        None,
    )
    if header_idx is None:
        raise RuntimeError(f"No 'IDS,' header found in {csv_path}")

    meta: dict[str, Any] = {}
    kv_pattern = re.compile(r"^\s*([^,]+)\s*,\s*(.*)\s*$")
    for line in lines[:header_idx]:
        s = line.strip()
        if not s or s.startswith("#") or s.lower().startswith("filename"):
            continue
        m = kv_pattern.match(s)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()

    for key in ("Weight", "Height", "BTPS", "BTPSE", "STPD", "Items", "Test no."):
        if key in meta and isinstance(meta[key], str):
            try:
                meta[key] = float(meta[key]) if "." in meta[key] else int(meta[key])
            except ValueError:
                pass

    df = pd.read_csv(
        StringIO("".join(lines[header_idx:])),
        header=0,
        engine="python",
        on_bad_lines="warn",
    )
    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) > 1 and re.fullmatch(r"\d+", str(df.columns[1])):
        df = df.rename(columns={df.columns[0]: "Type", df.columns[1]: "SampleRate"})
    else:
        df = df.rename(columns={df.columns[0]: "Type"})

    df = df[df["Type"].astype(str).str.upper() == "BXB"].reset_index(drop=True)

    if "hh:mm:ss" in df.columns:
        td = pd.to_timedelta(df["hh:mm:ss"].astype(str).str.strip(), errors="coerce")
        df["time_seconds"] = td.dt.total_seconds()
    elif "Item" in df.columns:
        item = pd.to_numeric(df["Item"], errors="coerce")
        df["time_seconds"] = item - item.iloc[0]
    else:
        raise RuntimeError(f"Cannot derive time_seconds for {csv_path}")

    skip = {"Type", "hh:mm:ss"}
    for col in df.columns:
        if col in skip:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, meta


# ---------------------------------------------------------------------------
# Power computation & marker extraction
# ---------------------------------------------------------------------------
def compute_metabolic_power(df: pd.DataFrame, weight_kg: float) -> pd.DataFrame:
    """Compute per-breath metabolic power (W/kg) using the Brockway equation.

    Fixes the latent bug in the legacy ``DataAnalysis/metabolic_analysis.py``
    which used ``K5_VO2`` for both VO2 and VCO2.

    Args:
        df: Output of :func:`parse_metabolics_csv`.
        weight_kg: Subject mass in kilograms (from ``meta['Weight']``).

    Returns:
        DataFrame with columns ``time_seconds``, ``K5_VO2``, ``K5_VCO2``,
        ``marker``, ``power_W_per_kg``, with NaN rows dropped.
    """
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be positive, got {weight_kg}")
    needed = {"K5_VO2", "K5_VCO2", "time_seconds"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "time_seconds": df["time_seconds"].astype(float),
            "K5_VO2": df["K5_VO2"].astype(float),
            "K5_VCO2": df["K5_VCO2"].astype(float),
            "marker": df.get("Marker", pd.Series(np.zeros(len(df))))
            .fillna(0)
            .astype(int),
        }
    )
    out["power_W_per_kg"] = (
        BROCKWAY_VO2_W_PER_ML_MIN * out["K5_VO2"]
        + BROCKWAY_VCO2_W_PER_ML_MIN * out["K5_VCO2"]
    ) / float(weight_kg)
    out = out.dropna(subset=["time_seconds", "K5_VO2", "K5_VCO2"]).reset_index(drop=True)
    return out


def extract_markers(
    df: pd.DataFrame,
    dedup_window_s: float = DEFAULT_MARKER_DEDUP_S,
) -> list[Marker]:
    """Extract marker times from a parsed metabolics DataFrame.

    Collapses double-presses: consecutive raw markers whose **pairwise** gap
    is at most ``dedup_window_s`` are merged into one burst; the **last**
    marker in each burst is kept (the later press is treated as the intended
    event).

    Args:
        df: Output of :func:`parse_metabolics_csv` (must contain ``Marker`` and
            ``time_seconds`` columns).
        dedup_window_s: If ``times[i] - times[i-1] <=`` this value, marker ``i``
            continues the current burst; otherwise a new burst starts.

    Returns:
        Ordered list of :class:`Marker`.
    """
    if "Marker" not in df.columns or "time_seconds" not in df.columns:
        return []
    mask = df["Marker"].fillna(0).astype(int) == 1
    times = df.loc[mask, "time_seconds"].astype(float).to_numpy()
    items = (
        df.loc[mask, "Item"].astype(float).to_numpy()
        if "Item" in df.columns
        else np.full(len(times), np.nan)
    )
    if len(times) == 0:
        return []

    # Chain bursts: gap between consecutive *raw* markers <= dedup_window_s
    # -> same burst; keep the last index in each burst.
    keep_idx: list[int] = []
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > dedup_window_s:
            keep_idx.append(i - 1)
    keep_idx.append(len(times) - 1)

    return [
        Marker(
            index=k,
            time_s=float(times[i]),
            item=(int(items[i]) if not np.isnan(items[i]) else None),
        )
        for k, i in enumerate(keep_idx)
    ]


# ---------------------------------------------------------------------------
# Trial windowing
# ---------------------------------------------------------------------------
def select_trial_windows(
    markers: list[Marker],
    trial_keys: Iterable[str] = TRIAL_KEYS,
    max_window_s: float = DEFAULT_MAX_WINDOW_S,
    manual_overrides: dict[str, dict[str, Any]] | None = None,
    session_end_s: float | None = None,
) -> list[TrialWindow]:
    """Pair markers to trial keys and produce per-trial time windows.

    Default behaviour pairs the first ``len(trial_keys)`` deduplicated markers
    to the canonical trial order. Each window ends at
    ``min(next_trial_start, start + max_window_s)`` (capped at 2:00 by default).

    Args:
        markers: Output of :func:`extract_markers` (deduplicated).
        trial_keys: Chronological trial keys to assign (default
            :data:`TRIAL_KEYS`). Restrict to those present in the
            ``experiment_order.json`` entry for the MIH.
        max_window_s: Hard cap on window duration from the start.
        manual_overrides: Per-trial-key overrides during analyst review.
            Each value may set:

            * ``start_s`` (float, seconds): absolute trial start; ignores
              markers entirely. Use this when no marker exists at the
              correct moment, e.g. ``{"baseline_1": {"start_s": 959.0}}``.
            * ``start_marker_idx`` (int): pick a different marker as the
              start. Ignored if ``start_s`` is also given.
            * ``end_s`` (float): absolute window end in seconds.
            * ``notes`` (str): free-form note carried into the result.
        session_end_s: Last valid time in the session, used as the upper bound
            on the final window.

    Returns:
        List of :class:`TrialWindow`. Synthetic starts (``start_s`` overrides)
        carry ``start_marker_idx == -1``.
    """
    overrides = manual_overrides or {}
    keys = list(trial_keys)
    n_keys = len(keys)

    # Resolve each trial's start time and marker index up-front so that the
    # window-end calculation can use the *next* trial's start regardless of
    # whether it is marker-based or synthetic.
    @dataclass(slots=True)
    class _Start:
        time_s: float | None
        marker_idx: int  # -1 for synthetic

    resolved: list[_Start] = []
    for k_i, key in enumerate(keys):
        ovr = overrides.get(key, {})
        if "start_s" in ovr:
            resolved.append(_Start(time_s=float(ovr["start_s"]), marker_idx=-1))
            continue
        if "start_marker_idx" in ovr:
            m_idx = int(ovr["start_marker_idx"])
        else:
            m_idx = k_i
        if 0 <= m_idx < len(markers):
            resolved.append(_Start(time_s=float(markers[m_idx].time_s), marker_idx=m_idx))
        else:
            resolved.append(_Start(time_s=None, marker_idx=m_idx))

    windows: list[TrialWindow] = []
    for k_i, (key, start) in enumerate(zip(keys, resolved)):
        if start.time_s is None:
            logger.warning(
                "Trial '%s' has no start (m_idx=%d, n_markers=%d) -- skipping",
                key, start.marker_idx, len(markers),
            )
            continue
        start_s = start.time_s
        ovr = overrides.get(key, {})

        if "end_s" in ovr:
            end_s = float(ovr["end_s"])
        else:
            # Find the next start time strictly after this one. Other trials'
            # starts that fall *before* this one are ignored (can happen with
            # start_s overrides) so the window doesn't collapse.
            next_start: float | None = None
            for j in range(k_i + 1, n_keys):
                t = resolved[j].time_s
                if t is not None and t > start_s:
                    next_start = t
                    break
            if next_start is None and start.marker_idx >= 0:
                for j in range(start.marker_idx + 1, len(markers)):
                    t = float(markers[j].time_s)
                    if t > start_s:
                        next_start = t
                        break
            cap = start_s + max_window_s
            if next_start is not None:
                end_s = min(cap, next_start)
            elif session_end_s is not None:
                end_s = min(cap, float(session_end_s))
            else:
                end_s = cap

        windows.append(
            TrialWindow(
                trial_key=key,
                start_marker_idx=start.marker_idx,
                start_s=start_s,
                end_s=float(end_s),
                duration_s=float(end_s - start_s),
                notes=ovr.get("notes"),
            )
        )
    return windows


def markers_with_trial_assignment(
    markers: list[Marker],
    trial_keys: Iterable[str],
    exp_order_for_mih: dict[str, Any] | None = None,
    manual_overrides: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return a per-marker table showing which trial+geometry each marker maps to.

    Default chronological pairing is ``markers[i]`` -> ``trial_keys[i]``. Any
    ``start_marker_idx`` override moves a trial onto a different marker.
    Markers with no trial assigned are tagged ``unused`` (likely "dud" presses).

    Args:
        markers: Deduplicated markers from :func:`extract_markers`.
        trial_keys: Ordered trial keys for this MIH (subset of :data:`TRIAL_KEYS`).
        exp_order_for_mih: ``experiment_order.json[mih_id]`` block; used to
            look up each trial's ``geometry`` and ``angles_alpha_beta_gamma``.
            If ``None``, geometry columns are left blank.
        manual_overrides: Same dict accepted by :func:`select_trial_windows`.

    Returns:
        DataFrame with columns
        ``marker_idx``, ``time_s``, ``item``,
        ``trial_key``, ``geometry``, ``angles``, ``override``.
    """
    overrides = manual_overrides or {}
    keys = list(trial_keys)

    # Map each trial to its assigned marker (skip trials with start_s override;
    # those don't claim a marker -- they get a synthetic row at the bottom).
    marker_to_trial: dict[int, str] = {}
    synthetic_starts: list[tuple[str, float]] = []
    for k_i, key in enumerate(keys):
        ovr = overrides.get(key, {})
        if "start_s" in ovr:
            synthetic_starts.append((key, float(ovr["start_s"])))
            continue
        m_idx = int(ovr["start_marker_idx"]) if "start_marker_idx" in ovr else k_i
        if 0 <= m_idx < len(markers):
            marker_to_trial[m_idx] = key

    def _meta(trial_key: str) -> tuple[str | None, list[float] | None]:
        if exp_order_for_mih is None:
            return None, None
        meta = exp_order_for_mih.get(trial_key, {}) or {}
        return meta.get("geometry"), meta.get("angles_alpha_beta_gamma")

    rows: list[dict[str, Any]] = []
    for m in markers:
        trial_key = marker_to_trial.get(m.index, "unused")
        geometry, angles = _meta(trial_key) if trial_key != "unused" else (None, None)
        ovr = overrides.get(trial_key, {})
        rows.append(
            {
                "marker_idx": m.index,
                "time_s": round(m.time_s, 1),
                "item": m.item,
                "trial_key": trial_key,
                "geometry": geometry,
                "angles": angles,
                "override": (
                    "start_marker_idx" in ovr if trial_key != "unused" else False
                ),
            }
        )

    for trial_key, t_s in synthetic_starts:
        geometry, angles = _meta(trial_key)
        rows.append(
            {
                "marker_idx": -1,
                "time_s": round(t_s, 1),
                "item": None,
                "trial_key": trial_key,
                "geometry": geometry,
                "angles": angles,
                "override": True,
            }
        )

    return pd.DataFrame(rows)


def prepare_session_review(
    df_power: pd.DataFrame,
    markers: list[Marker],
    data_root: Path,
    mih_id: str,
    manual_overrides: dict[str, dict[str, Any]] | None = None,
    max_window_s: float = DEFAULT_MAX_WINDOW_S,
) -> tuple[dict[str, Any], list[str], list[TrialWindow]]:
    """Load experiment order and build trial windows (notebook / REPL helper).

    Call :func:`plot_windows_overlay` with the returned ``windows`` to re-view
    the full metabolic trace after editing ``manual_overrides``.

    Args:
        df_power: Power time series (for session end time).
        markers: Deduplicated markers from :func:`extract_markers`.
        data_root: Path to ``data_V2``.
        mih_id: Participant id, e.g. ``"MIH01"``.
        manual_overrides: Per-trial overrides for :func:`select_trial_windows`.
        max_window_s: Cap on window length from each start marker.

    Returns:
        Tuple ``(exp_order, trial_keys, windows)``. Use ``exp_order[mih_id]``
        for per-trial metadata; ``trial_keys`` is the ordered key list used.
    """
    exp_order = load_experiment_order(data_root)
    if mih_id not in exp_order:
        raise KeyError(f"{mih_id} not in experiment_order.json")
    trial_keys = [k for k in TRIAL_KEYS if k in exp_order[mih_id]]
    session_end = float(df_power["time_seconds"].max()) if len(df_power) else None
    windows = select_trial_windows(
        markers,
        trial_keys=trial_keys,
        max_window_s=max_window_s,
        manual_overrides=manual_overrides,
        session_end_s=session_end,
    )
    return exp_order, trial_keys, windows


# ---------------------------------------------------------------------------
# Steady-state estimation
# ---------------------------------------------------------------------------
def _exp_rise(t: np.ndarray, y_ss: float, tau: float) -> np.ndarray:
    return y_ss * (1.0 - np.exp(-t / tau))


def _last_n_avg(
    t: np.ndarray,
    y: np.ndarray,
    frac: float,
    reason: str,
) -> SSResult:
    """Fallback steady-state estimator: mean of the last ``frac`` of points."""
    if len(y) == 0:
        return SSResult(
            y_ss=float("nan"), y_ss_std=float("nan"),
            tau=float("nan"), tau_std=float("nan"),
            r_squared=float("nan"), n_points=0,
            fit_method=f"empty:{reason}", residual_std=float("nan"),
        )
    k = max(int(frac * len(y)), 1)
    tail = y[-k:]
    y_ss = float(np.mean(tail))
    sd = float(np.std(tail, ddof=1)) if k > 1 else 0.0
    sem = sd / math.sqrt(k) if k > 1 else 0.0
    return SSResult(
        y_ss=y_ss,
        y_ss_std=sem,
        tau=float("nan"),
        tau_std=float("nan"),
        r_squared=float("nan"),
        n_points=int(len(t)),
        fit_method=f"last{int(frac * 100)}pct_avg({reason})",
        residual_std=sd,
    )


def estimate_steady_state(
    time_s: np.ndarray,
    power_W_per_kg: np.ndarray,
    tau_seed: float = DEFAULT_TAU_S,
) -> SSResult:
    """Estimate steady-state metabolic power via exponential-rise fit.

    Model: ``y = y_ss * (1 - exp(-t / tau))`` with time normalised so the
    window starts at 0. Uncertainty on ``y_ss`` is the square-root of the
    diagonal of the covariance matrix returned by ``scipy.optimize.curve_fit``.

    Falls back to :func:`_last_n_avg` (last 30% of points) only when the fit
    fails (or when there are too few valid points to fit).

    Args:
        time_s: Time array (s) for the window.
        power_W_per_kg: Per-breath metabolic power (W/kg) for the window.
        tau_seed: Initial guess for tau (s).
    Returns:
        :class:`SSResult` with point estimate, std and fit diagnostics.
    """
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(power_W_per_kg, dtype=float)
    valid = ~(np.isnan(t) | np.isnan(y))
    t = t[valid]
    y = y[valid]
    if len(t) == 0:
        return SSResult(
            y_ss=float("nan"), y_ss_std=float("nan"),
            tau=float("nan"), tau_std=float("nan"),
            r_squared=float("nan"), n_points=0,
            fit_method="empty", residual_std=float("nan"),
        )

    # Normalise time to start at 0.
    t = t - t[0]

    # With too few points, a 2-parameter nonlinear fit is underdetermined.
    if len(t) < 3:
        return _last_n_avg(t, y, frac=0.30, reason=f"too_few_points={len(t)}")

    try:
        tail_n = max(int(0.3 * len(y)), 5)
        y_ss_guess = float(np.mean(y[-tail_n:]))
        bounds = ([0.5, 5.0], [50.0, 300.0])
        popt, pcov = curve_fit(
            _exp_rise,
            t, y,
            p0=[y_ss_guess, tau_seed],
            bounds=bounds,
            maxfev=10000,
        )
        y_ss = float(popt[0])
        tau = float(popt[1])
        with np.errstate(invalid="ignore"):
            perr = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        y_ss_std = float(perr[0])
        tau_std = float(perr[1])

        y_hat = _exp_rise(t, y_ss, tau)
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        return SSResult(
            y_ss=y_ss, y_ss_std=y_ss_std,
            tau=tau, tau_std=tau_std,
            r_squared=r2, n_points=int(len(t)),
            fit_method="exponential_rise",
            residual_std=residual_std,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("Exp fit failed (%s); falling back to last-30%% mean", exc)
        return _last_n_avg(t, y, frac=0.30, reason=f"fit_failed:{type(exc).__name__}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_full_session(
    df_power: pd.DataFrame,
    markers: list[Marker],
    title: str,
) -> go.Figure:
    """Plot full-session power-vs-time with all markers as labelled vlines."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_power["time_seconds"],
            y=df_power["power_W_per_kg"],
            mode="lines",
            name="Metabolic Power",
            line=dict(color="lightblue", width=1),
            opacity=0.7,
        )
    )
    if len(df_power) > 15:
        smooth = df_power["power_W_per_kg"].rolling(
            window=15, min_periods=1, center=True,
        ).mean()
        fig.add_trace(
            go.Scatter(
                x=df_power["time_seconds"],
                y=smooth,
                mode="lines",
                name="Smoothed (15-pt)",
                line=dict(color="steelblue", width=2),
            )
        )
    for m in markers:
        fig.add_vline(
            x=m.time_s,
            line_dash="dash", line_color="crimson", opacity=0.6,
            annotation_text=f"M{m.index}",
            annotation_position="top",
        )
    fig.update_layout(
        title=f"{title} - Full session ({len(markers)} markers)",
        xaxis_title="Time (s)",
        yaxis_title="Metabolic Power (W/kg)",
        hovermode="x unified",
        width=1200,
        height=500,
    )
    return fig


def plot_windows_overlay(
    df_power: pd.DataFrame,
    markers: list[Marker],
    windows: list[TrialWindow],
    title: str,
) -> go.Figure:
    """Plot full session with chosen trial windows shaded for analyst review."""
    fig = plot_full_session(df_power, markers, title)
    for w in windows:
        if w.start_marker_idx == -1:
            label = f"{w.trial_key} (t={w.start_s:.0f}s, manual)"
        else:
            label = f"{w.trial_key} (m#{w.start_marker_idx})"
        fig.add_vrect(
            x0=w.start_s, x1=w.end_s,
            fillcolor="green", opacity=0.12,
            line_width=0, layer="below",
            annotation_text=label,
            annotation_position="top left",
        )
    fig.update_layout(title=f"{title} - Trial windows ({len(windows)})")
    return fig


def plot_trial_window(
    df_power: pd.DataFrame,
    window: TrialWindow,
    ss: SSResult,
    title_prefix: str = "",
) -> go.Figure:
    """Plot a single trial window with raw data, exp-fit overlay and y_ss line."""
    mask = (df_power["time_seconds"] >= window.start_s) & (
        df_power["time_seconds"] <= window.end_s
    )
    seg = df_power.loc[mask].copy()
    seg["t_rel"] = seg["time_seconds"] - window.start_s

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seg["t_rel"],
            y=seg["power_W_per_kg"],
            mode="markers+lines",
            name="Raw",
            marker=dict(size=4, opacity=0.6, color="steelblue"),
            line=dict(color="lightblue", width=1),
        )
    )
    if ss.fit_method == "exponential_rise" and not math.isnan(ss.tau):
        t_max = float(seg["t_rel"].max()) if len(seg) else 1.0
        t_grid = np.linspace(0.0, max(t_max, 1.0), 200)
        fig.add_trace(
            go.Scatter(
                x=t_grid,
                y=_exp_rise(t_grid, ss.y_ss, ss.tau),
                mode="lines",
                name=f"Exp fit (tau={ss.tau:.1f}s)",
                line=dict(color="crimson", width=2),
            )
        )
    if not math.isnan(ss.y_ss):
        fig.add_hline(
            y=ss.y_ss,
            line_dash="dash",
            line_color="green",
            annotation_text=f"y_ss = {ss.y_ss:.3f} +/- {ss.y_ss_std:.3f} W/kg",
        )
    r2_txt = "n/a" if math.isnan(ss.r_squared) else f"{ss.r_squared:.3f}"
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=(
            f"R^2 = {r2_txt}<br>"
            f"n = {ss.n_points}<br>"
            f"method = {ss.fit_method}"
        ),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="black", borderwidth=1,
    )
    fig.update_layout(
        title=f"{title_prefix}{window.trial_key} - duration {window.duration_s:.1f}s",
        xaxis_title="Time since marker (s)",
        yaxis_title="Power (W/kg)",
        width=900, height=450,
    )
    return fig


def _optional_plot_y(v: float | None) -> float | None:
    """Convert a value to float for Plotly, or ``None`` if missing / NaN (gaps)."""
    if v is None or math.isnan(v):
        return None
    return float(v)


def plot_cot_raw_vs_linear_baseline(
    per_trial: dict[str, CoTResult],
    title_prefix: str = "",
    trial_keys: Iterable[str] | None = None,
) -> go.Figure:
    """Plot raw CoT vs linear baseline-adjusted CoT across trials.

    The x axis is ``0 .. N-1`` in session order; tick labels are the trial keys
    (e.g. ``baseline_1``, ``\"1\"``, …, ``baseline_2``). Intended for use after
    :func:`apply_baseline_corrections` so ``cot_linear_baseline_adj`` is filled.

    Args:
        per_trial: Mapping from trial key to :class:`CoTResult`.
        title_prefix: Text prepended to the plot title (e.g. ``\"MIH01: \"``).
        trial_keys: Ordered keys to include. Default: :data:`TRIAL_KEYS` in
            order, intersected with ``per_trial``.

    Returns:
        Plotly figure (caller may ``.show()`` or write HTML).

    Example:
        >>> fig = plot_cot_raw_vs_linear_baseline(results, title_prefix="MIH01: ")
    """
    keys = (
        list(trial_keys)
        if trial_keys is not None
        else [k for k in TRIAL_KEYS if k in per_trial]
    )
    x_idx = list(range(len(keys)))
    y_raw = [_optional_plot_y(per_trial[k].cot_raw) for k in keys]
    y_lin = [_optional_plot_y(per_trial[k].cot_linear_baseline_adj) for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=y_raw,
            mode="lines+markers",
            name="CoT raw",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx,
            y=y_lin,
            mode="lines+markers",
            name="CoT linear baseline adj",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"{title_prefix}cost of transport — raw vs linear baseline adjustment",
        xaxis_title="Trial index",
        xaxis=dict(tickmode="array", tickvals=x_idx, ticktext=keys),
        yaxis_title="CoT (J·kg⁻¹·m⁻¹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60),
        width=900,
        height=450,
    )
    return fig


def plot_cot_power_velocity_breakdown(
    per_trial: dict[str, CoTResult],
    title_prefix: str = "",
    trial_keys: Iterable[str] | None = None,
) -> go.Figure:
    """Stacked subplots: CoT / steady-state power / velocity per trial.

    CoT is mathematically ``power / velocity``, so a high-CoT trial is either
    "high metabolic power" or "low velocity" (or both). This view stacks the
    three series on a shared x axis (trial index) to make the source of any
    CoT spike immediately readable.

    Args:
        per_trial: Mapping from trial key to :class:`CoTResult`.
        title_prefix: Text prepended to the figure title (e.g. ``"MIH01: "``).
        trial_keys: Ordered keys to include. Default: :data:`TRIAL_KEYS`
            intersected with ``per_trial``.

    Returns:
        Plotly figure with three rows (CoT raw + linear, power, velocity).
    """
    keys = (
        list(trial_keys)
        if trial_keys is not None
        else [k for k in TRIAL_KEYS if k in per_trial]
    )
    x_idx = list(range(len(keys)))
    cot_raw = [_optional_plot_y(per_trial[k].cot_raw) for k in keys]
    cot_lin = [_optional_plot_y(per_trial[k].cot_linear_baseline_adj) for k in keys]
    power = [_optional_plot_y(per_trial[k].steady_state_W_per_kg) for k in keys]
    velocity = [_optional_plot_y(per_trial[k].velocity_m_s) for k in keys]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Cost of transport (J·kg⁻¹·m⁻¹)",
            "Steady-state metabolic power (W·kg⁻¹)",
            "Walking velocity (m·s⁻¹)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=x_idx, y=cot_raw, mode="lines+markers", name="CoT raw"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx, y=cot_lin, mode="lines+markers",
            name="CoT linear baseline adj", line=dict(dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx, y=power, mode="lines+markers",
            name="Steady-state power", line=dict(color="firebrick"),
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_idx, y=velocity, mode="lines+markers",
            name="Velocity", line=dict(color="seagreen"),
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=x_idx, ticktext=keys,
        row=3, col=1, title_text="Trial",
    )
    fig.update_layout(
        title=f"{title_prefix}CoT decomposition — power × velocity",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        margin=dict(t=80),
        height=750,
        width=950,
    )
    return fig


def plot_power_vs_velocity_iso_cot(
    per_trial: dict[str, CoTResult],
    title_prefix: str = "",
    trial_keys: Iterable[str] | None = None,
    cot_source: Literal["raw", "linear"] = "linear",
    iso_cot_lines: Iterable[float] | None = None,
) -> go.Figure:
    """Scatter of (effective) power vs velocity with iso-CoT reference lines.

    The plot enforces the identity ``power = CoT · velocity``. For each trial
    the y-coordinate is therefore ``cot · velocity`` where ``cot`` is selected
    by ``cot_source``:

    * ``"raw"``: uses ``cot_raw``, so ``y == steady_state_W_per_kg``.
    * ``"linear"``: uses the precomputed ``cot_linear_baseline_adj`` from
      :func:`apply_baseline_corrections` (no recomputation here). With this
      choice ``baseline_1`` and ``baseline_2`` collapse to the same iso-CoT
      line by construction.

    Args:
        per_trial: Mapping from trial key to :class:`CoTResult`. Must already
            have ``cot_linear_baseline_adj`` filled when ``cot_source="linear"``
            (call :func:`apply_baseline_corrections` first).
        title_prefix: Text prepended to the title (e.g. ``"MIH01: "``).
        trial_keys: Ordered keys to include. Default: :data:`TRIAL_KEYS`
            intersected with ``per_trial``.
        cot_source: ``"raw"`` (default ``"linear"``) selects which CoT to plot
            and to derive iso-lines from.
        iso_cot_lines: Override CoT values (J·kg⁻¹·m⁻¹) for the reference rays.
            Defaults to 5 evenly spaced lines spanning the observed CoT range
            for the chosen ``cot_source``.

    Returns:
        Plotly figure (caller may ``.show()`` or write HTML).

    Raises:
        ValueError: If ``cot_source`` is unknown, no plottable trials exist,
            or ``cot_source="linear"`` but no trial has a finite
            ``cot_linear_baseline_adj``.
    """
    if cot_source not in ("raw", "linear"):
        raise ValueError(
            f"Unsupported cot_source={cot_source!r}; expected 'raw' or 'linear'"
        )

    keys = (
        list(trial_keys)
        if trial_keys is not None
        else [k for k in TRIAL_KEYS if k in per_trial]
    )

    def _trial_cot(r: CoTResult) -> float:
        if cot_source == "raw":
            return float(r.cot_raw)
        adj = r.cot_linear_baseline_adj
        return float(adj) if adj is not None else math.nan

    rows: list[tuple[str, float, float, float, float]] = []
    for k in keys:
        r = per_trial[k]
        c = _trial_cot(r)
        if math.isnan(r.velocity_m_s) or math.isnan(c):
            continue
        rows.append(
            (k, float(r.velocity_m_s), float(r.cot_raw), float(r.steady_state_W_per_kg), c)
        )

    if not rows:
        if cot_source == "linear":
            raise ValueError(
                "No trials with finite cot_linear_baseline_adj. "
                "Did you forget to call apply_baseline_corrections()?"
            )
        raise ValueError("No trials with finite velocity/CoT to plot")

    labels = [r[0] for r in rows]
    vel = np.asarray([r[1] for r in rows], dtype=float)
    cot_raw = np.asarray([r[2] for r in rows], dtype=float)
    pwr_raw = np.asarray([r[3] for r in rows], dtype=float)
    cot = np.asarray([r[4] for r in rows], dtype=float)

    pwr_eff = cot * vel

    if iso_cot_lines is None:
        cot_min, cot_max = float(cot.min()), float(cot.max())
        iso = [cot_min] if cot_min == cot_max else list(np.linspace(cot_min, cot_max, 5))
    else:
        iso = list(iso_cot_lines)

    cot_label = "CoT raw" if cot_source == "raw" else "CoT linear baseline adj"
    cot_units = "J·kg⁻¹·m⁻¹"
    y_axis_title = (
        "Steady-state power (W·kg⁻¹)"
        if cot_source == "raw"
        else "Effective power = CoT_linear · velocity (W·kg⁻¹)"
    )

    hover_text: list[str] = []
    for i, k in enumerate(labels):
        lines = [
            f"trial={k}",
            f"velocity={vel[i]:.3f} m/s",
            f"{cot_label}={cot[i]:.3f} {cot_units}",
            f"effective power={pwr_eff[i]:.3f} W/kg",
        ]
        if cot_source == "linear":
            lines.append(f"raw power={pwr_raw[i]:.3f} W/kg  (CoT raw={cot_raw[i]:.3f})")
        hover_text.append("<br>".join(lines))

    v_max = float(vel.max()) * 1.08
    v_grid = np.linspace(0.0, v_max, 50)

    fig = go.Figure()
    for c in iso:
        fig.add_trace(
            go.Scatter(
                x=v_grid,
                y=c * v_grid,
                mode="lines",
                line=dict(color="lightgray", dash="dot", width=1),
                hovertemplate=f"iso-CoT = {c:.2f} {cot_units}<extra></extra>",
                name=f"CoT = {c:.2f}",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=v_max,
            y=c * v_max,
            text=f"CoT={c:.2f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
        )

    fig.add_trace(
        go.Scatter(
            x=vel,
            y=pwr_eff,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=12,
                color=cot,
                colorscale="Viridis",
                colorbar=dict(title=f"{cot_label} ({cot_units})"),
                line=dict(color="black", width=0.5),
            ),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            name="trials",
        )
    )

    fig.update_layout(
        title=(
            f"{title_prefix}power vs velocity "
            f"(iso-CoT lines, {cot_label.lower()})"
        ),
        xaxis_title="Velocity (m·s⁻¹)",
        yaxis_title=y_axis_title,
        xaxis=dict(range=[0.0, v_max]),
        yaxis=dict(rangemode="tozero"),
        showlegend=False,
        width=850,
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Cost of transport & baseline corrections
# ---------------------------------------------------------------------------
def compute_cot(
    ss: SSResult,
    walking_time_s: float,
    distance_m: float,
    trial_key: str,
    geometry: str | None,
    marker_time_s: float,
    window_end_s: float,
    notes: str | None = None,
) -> CoTResult:
    """Compute Cost of Transport (J / (kg * m)) from a steady-state result.

    Args:
        ss: Steady-state estimate for the trial window.
        walking_time_s: Active walking time for the trial (from
            ``experiment_order.json`` ``walking_time_s`` field).
        distance_m: Total distance walked (from ``total_distance_m``).
        trial_key: e.g. ``"baseline_1"``, ``"3"``.
        geometry: Crutch geometry string, e.g. ``"G0_A95_B95"``.
        marker_time_s: Window start (s, absolute, in session time).
        window_end_s: Window end (s, absolute).
        notes: Free-form note (e.g. fallback messages).

    Returns:
        :class:`CoTResult` with raw CoT and uncertainty (baseline corrections
        get filled in later by :func:`apply_baseline_corrections`).
    """
    if walking_time_s <= 0 or distance_m <= 0 or math.isnan(ss.y_ss):
        velocity = float("nan")
        cot = float("nan")
        cot_std = float("nan")
    else:
        velocity = float(distance_m) / float(walking_time_s)
        cot = ss.y_ss / velocity
        # Relative std of CoT == relative std of y_ss when distance/time are
        # treated as exact (placeholder until we wire in walking_time_std_s).
        if ss.y_ss > 0 and not math.isnan(ss.y_ss_std):
            cot_std = abs(cot) * (ss.y_ss_std / ss.y_ss)
        else:
            cot_std = float("nan")

    return CoTResult(
        trial_key=trial_key,
        geometry=geometry,
        marker_time_s=float(marker_time_s),
        window_end_s=float(window_end_s),
        walking_time_s=float(walking_time_s),
        distance_m=float(distance_m),
        velocity_m_s=float(velocity),
        steady_state_W_per_kg=float(ss.y_ss),
        steady_state_std=float(ss.y_ss_std),
        exp_fit_tau_s=float(ss.tau),
        exp_fit_r2=float(ss.r_squared),
        cot_raw=float(cot),
        cot_raw_std=float(cot_std),
        cot_linear_baseline_adj=None,
        cot_ratio_baseline_norm=None,
        fit_method=ss.fit_method,
        notes=notes,
    )


def apply_baseline_corrections(
    per_trial: dict[str, CoTResult],
) -> dict[str, CoTResult]:
    """Compute baseline-corrected CoT variants in place.

    Adds two extra fields on every trial (including the baselines themselves):

    * ``cot_linear_baseline_adj`` -- subtract a linearly interpolated baseline
      drift, then recentre on the mean baseline:
      ``cot - bl_pred(i) + mean(b1, b2)``.
    * ``cot_ratio_baseline_norm`` -- ``cot / bl_pred(i)`` (1.0 == on-baseline).

    where ``bl_pred(i) = b1 + (b2 - b1) * (i / (n_total - 1))`` and ``i`` is
    the trial's chronological position.

    Args:
        per_trial: dict keyed by trial key.

    Returns:
        The same ``per_trial`` dict, mutated in place.
    """
    if "baseline_1" not in per_trial or "baseline_2" not in per_trial:
        logger.warning("Missing baseline_1/baseline_2 -- skipping corrections")
        return per_trial

    b1 = per_trial["baseline_1"].cot_raw
    b2 = per_trial["baseline_2"].cot_raw
    if math.isnan(b1) or math.isnan(b2):
        logger.warning("Baseline CoT is NaN -- skipping corrections")
        return per_trial

    middle = [k for k in per_trial.keys() if k not in ("baseline_1", "baseline_2")]
    ordered = ["baseline_1", *middle, "baseline_2"]
    n_total = len(ordered)
    mean_bl = 0.5 * (b1 + b2)

    for i, key in enumerate(ordered):
        bl_pred = b1 + (b2 - b1) * (i / max(n_total - 1, 1))
        cot = per_trial[key].cot_raw
        if math.isnan(cot) or bl_pred <= 0:
            continue
        per_trial[key].cot_linear_baseline_adj = float(cot - bl_pred + mean_bl)
        per_trial[key].cot_ratio_baseline_norm = float(cot / bl_pred)

    return per_trial


# ---------------------------------------------------------------------------
# Orchestrator & save
# ---------------------------------------------------------------------------
def process_mih(
    mih_id: str,
    data_root: Path,
    exp_order: dict[str, dict[str, Any]],
    manual_overrides: dict[str, dict[str, Any]] | None = None,
    save_plots_dir: Path | None = None,
    tau_seed: float = DEFAULT_TAU_S,
    max_window_s: float = DEFAULT_MAX_WINDOW_S,
    marker_dedup_s: float = DEFAULT_MARKER_DEDUP_S,
) -> dict[str, Any]:
    """Run the full metabolics pipeline for one MIH and return its result block.

    Args:
        mih_id: e.g. ``"MIH01"``.
        data_root: Path to ``data_V2/``.
        exp_order: Loaded ``experiment_order.json`` dict.
        manual_overrides: See :func:`select_trial_windows`.
        save_plots_dir: If given, writes per-trial HTML plots there.
        tau_seed: Exp-fit tau initial guess.
        max_window_s: Window cap.
        marker_dedup_s: Dedup window for double-presses.

    Returns:
        Dict ready to be inserted under ``mih_id`` in the output JSON.
    """
    if mih_id not in exp_order:
        raise KeyError(f"{mih_id} not in experiment_order.json")
    mih_order = exp_order[mih_id]

    csv_path = find_metabolics_csv(data_root, mih_id)
    df_raw, meta = parse_metabolics_csv(csv_path)
    weight_kg = float(meta["Weight"])
    df_power = compute_metabolic_power(df_raw, weight_kg=weight_kg)
    markers = extract_markers(df_raw, dedup_window_s=marker_dedup_s)

    trial_keys = [k for k in TRIAL_KEYS if k in mih_order]
    windows = select_trial_windows(
        markers,
        trial_keys=trial_keys,
        max_window_s=max_window_s,
        manual_overrides=manual_overrides,
        session_end_s=float(df_power["time_seconds"].max()) if len(df_power) else None,
    )

    per_trial: dict[str, CoTResult] = {}
    for w in windows:
        seg = df_power[
            (df_power["time_seconds"] >= w.start_s)
            & (df_power["time_seconds"] <= w.end_s)
        ]
        ss = estimate_steady_state(
            seg["time_seconds"].to_numpy(),
            seg["power_W_per_kg"].to_numpy(),
            tau_seed=tau_seed,
        )
        trial_meta = mih_order.get(w.trial_key, {}) or {}
        wt_field = trial_meta.get("walking_time_s")
        if wt_field is None:
            walking_time_s = w.duration_s
            note = (
                "walking_time_s missing in experiment_order.json; "
                "using window duration as fallback"
            )
            logger.warning("[%s] %s: %s", mih_id, w.trial_key, note)
        else:
            walking_time_s = float(wt_field)
            note = w.notes
        distance_m = float(trial_meta.get("total_distance_m") or 0.0)
        cot = compute_cot(
            ss=ss,
            walking_time_s=walking_time_s,
            distance_m=distance_m,
            trial_key=w.trial_key,
            geometry=trial_meta.get("geometry"),
            marker_time_s=w.start_s,
            window_end_s=w.end_s,
            notes=note,
        )
        per_trial[w.trial_key] = cot

        if save_plots_dir is not None:
            save_plots_dir.mkdir(parents=True, exist_ok=True)
            fig = plot_trial_window(df_power, w, ss, title_prefix=f"{mih_id} ")
            fig.write_html(str(save_plots_dir / f"{mih_id}_{w.trial_key}.html"))

    apply_baseline_corrections(per_trial)

    block: dict[str, Any] = {
        "metabolics_csv": _relpath_under(csv_path, Path(data_root).parent),
        "weight_kg": weight_kg,
        "subject": {
            k: meta.get(k)
            for k in ("Firstname", "Lastname", "Height", "Gender", "Birth date")
            if k in meta
        },
        "tau_seed_s": tau_seed,
        "max_window_s": max_window_s,
        "marker_dedup_s": marker_dedup_s,
        "n_markers_raw": int(df_raw["Marker"].fillna(0).sum())
        if "Marker" in df_raw.columns
        else 0,
        "n_markers_dedup": len(markers),
    }
    valid_count = 0
    for key in trial_keys:
        if key in per_trial:
            entry = asdict(per_trial[key])
            block[key] = entry
            if not math.isnan(entry["cot_raw"]):
                valid_count += 1
        else:
            block[key] = None

    b1 = per_trial.get("baseline_1")
    b2 = per_trial.get("baseline_2")
    drift_pct: float | None = None
    if (
        b1 is not None
        and b2 is not None
        and not math.isnan(b1.cot_raw)
        and not math.isnan(b2.cot_raw)
        and (b1.cot_raw + b2.cot_raw) > 0
    ):
        drift_pct = 100.0 * (b2.cot_raw - b1.cot_raw) / (0.5 * (b1.cot_raw + b2.cot_raw))

    block["_summary"] = {
        "n_trial_keys": len(trial_keys),
        "n_valid": valid_count,
        "n_failed": len(trial_keys) - valid_count,
        "baseline_drift_pct": drift_pct,
        "fit_method_default": "exponential_rise",
    }
    return block


def _relpath_under(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` if possible, else its absolute string."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def _nan_to_none(obj: Any) -> Any:
    """Recursively convert NaN floats to ``None`` so json.dump emits ``null``."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(x) for x in obj]
    return obj


def save_results(
    all_mih_results: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write the per-MIH grouped JSON to ``output_path``.

    Args:
        all_mih_results: Mapping of ``mih_id -> result block`` (output of
            :func:`process_mih`).
        output_path: Destination, typically ``data_V2/metabolics_results.json``.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _nan_to_none(all_mih_results)
    output_path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    logger.info("Wrote %d MIH blocks to %s", len(all_mih_results), output_path)
    return output_path


def load_experiment_order(data_root: Path) -> dict[str, dict[str, Any]]:
    """Load ``data_V2/experiment_order.json``."""
    path = Path(data_root) / "experiment_order.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data_V2"
    exp_order = load_experiment_order(data_root)

    mih_id = "MIH01"
    out_dir = (
        find_metabolics_csv(data_root, mih_id).parent / "metabolics_analysis"
    )
    block = process_mih(
        mih_id=mih_id,
        data_root=data_root,
        exp_order=exp_order,
        save_plots_dir=out_dir,
    )
    print(json.dumps({mih_id: _nan_to_none(block)}, indent=2)[:2000])
