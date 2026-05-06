"""IMU per-trial review utilities.

Companion to :mod:`metabolics`. For each MIH session, this module:

1. Locates the on-disk IMU directory.
2. Maps every per-trial IMU CSV (one file per experiment) to the canonical
   trial keys used in ``experiment_order.json`` and ``metabolics_results.json``.
3. Renders a 3-row stacked Plotly figure (``acc_x``, ``acc_z``, ``force``)
   for each trial so the analyst can read off the actual walking start/stop
   time visually.

No automatic motion-onset / motion-end detection is done here -- the user
eyeballs the figures and updates ``walking_time_s`` in
``data_V2/experiment_order.json`` (or via a ``manual_overrides`` dict that
the metabolics pipeline already understands).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# We intentionally re-use the same MIH-id alias logic as ``metabolics`` so
# ``MIH02`` and ``MIH2`` resolve to the same on-disk directory.
from metabolics import _mih_id_aliases

logger = logging.getLogger(__name__)

#: Pattern matching the trailing ``_HHMMSS`` token in an IMU filename.
_HHMMSS_RE = re.compile(r"_(\d{6})$")

#: Time-column candidates seen across recordings.
#:
#: Older firmware called the column ``time``; current firmware calls it
#: ``acc_x_time``. Either way, the *values* in that column are NOT a reliable
#: real-time axis -- ``device-manager.js`` writes ``acc_x_time = sampleIdx * 5``
#: (commented "5 ms intervals, 200 Hz") but BLE notifications are actually
#: delivered at ~100 Hz, so the stored timestamps are ~2x too small. We
#: therefore reconstruct ``time_s`` from the sample index using a configurable
#: rate (see :data:`DEFAULT_SAMPLE_RATE_HZ`).
TIME_COLUMN_CANDIDATES: tuple[str, ...] = ("acc_x_time", "time")
DEFAULT_IMU_COLUMNS: tuple[str, str, str] = (
    "acc_x_data",
    "acc_z_data",
    "force",
)

#: Real BLE notify rate (Hz). Verified empirically: MIH01 baseline_1 has
#: 12 125 samples and the trial took ~120 s (12 125 / 100 = 121 s).
DEFAULT_SAMPLE_RATE_HZ: float = 100.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_imu_dir(data_root: Path, mih_id: str) -> Path | None:
    """Locate the IMU directory for ``mih_id`` under ``data_root/raw``.

    Args:
        data_root: Path to ``data_V2`` (the directory that contains ``raw/``).
        mih_id: Subject identifier, e.g. ``"MIH01"``. Also accepts the
            unpadded variant (``"MIH02"`` <-> ``"MIH2"``).

    Returns:
        The IMU directory if it exists, else ``None`` (some sessions have no
        IMU recording -- e.g. ``MIH16``).
    """
    raw_root = Path(data_root) / "raw"
    base: Path | None = next(
        (raw_root / a for a in _mih_id_aliases(mih_id) if (raw_root / a).is_dir()),
        None,
    )
    if base is None:
        return None
    sessions = sorted(p for p in base.iterdir() if p.is_dir())
    if not sessions:
        return None
    session = sessions[0]
    for d in ("IMU", "imu"):
        candidate = session / d
        if candidate.is_dir():
            return candidate
    return None


def list_trial_imu_files(
    imu_dir: Path,
    trial_keys: Iterable[str],
    manual_overrides: dict[str, str | Path] | None = None,
) -> dict[str, Path | None]:
    """Pair each trial key with one IMU CSV.

    Default behaviour: list non-``opencap_events`` CSVs, sort them by the
    trailing ``_HHMMSS`` token in the filename (chronological), and pair the
    *i*-th file with ``trial_keys[i]``. Any chronological mismatch (extra or
    missing files) is logged and the analyst is expected to fix it via
    ``manual_overrides``.

    Args:
        imu_dir: Path to the IMU directory.
        trial_keys: Ordered trial keys for the MIH (e.g.
            :data:`metabolics.TRIAL_KEYS` filtered to those present in
            ``experiment_order.json[mih_id]``).
        manual_overrides: Per-trial filename overrides. Each value may be a
            relative filename, an absolute path, or a substring that matches
            exactly one file in the directory.

    Returns:
        Dict ``{trial_key -> Path | None}``. ``None`` indicates that the
        chronological pairing ran out of files for that key.
    """
    overrides = manual_overrides or {}
    keys = list(trial_keys)

    all_csvs = sorted(p for p in imu_dir.glob("*.csv") if "_opencap_events" not in p.name)

    def _hhmmss(p: Path) -> str:
        m = _HHMMSS_RE.search(p.stem)
        return m.group(1) if m else p.stem

    chrono = sorted(all_csvs, key=_hhmmss)

    if len(chrono) != len(keys):
        logger.warning(
            "%s: %d IMU files but %d trial keys -- chronological pairing may "
            "be off; use manual_overrides to fix.",
            imu_dir, len(chrono), len(keys),
        )

    mapping: dict[str, Path | None] = {}
    for i, key in enumerate(keys):
        mapping[key] = chrono[i] if i < len(chrono) else None

    for key, ovr in overrides.items():
        path = _resolve_override(imu_dir, ovr)
        if path is None:
            logger.warning("%s: override %r for %s did not resolve", imu_dir, ovr, key)
        mapping[key] = path

    return mapping


def _resolve_override(imu_dir: Path, ovr: str | Path) -> Path | None:
    """Resolve a manual_imu_overrides value to a concrete file path."""
    p = Path(ovr) if not isinstance(ovr, Path) else ovr
    if p.is_absolute() and p.is_file():
        return p
    direct = imu_dir / p.name
    if direct.is_file():
        return direct
    matches = [f for f in imu_dir.glob("*.csv") if str(ovr) in f.name and "_opencap_events" not in f.name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning("%s: override %r is ambiguous (%d matches): %s",
                       imu_dir, ovr, len(matches), [m.name for m in matches])
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_imu_csv(
    csv_path: Path,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Parse a per-trial IMU CSV.

    A real-time ``time_s`` axis is reconstructed from the sample index using
    ``sample_rate_hz`` (default :data:`DEFAULT_SAMPLE_RATE_HZ` = 100 Hz). The
    raw ``acc_x_time`` / ``time`` column is preserved but **not** trusted --
    see the :data:`TIME_COLUMN_CANDIDATES` docstring for why.

    Args:
        csv_path: Path to the IMU CSV.
        sample_rate_hz: Real BLE delivery rate. Tweak only if a session was
            recorded with a different firmware.

    Returns:
        DataFrame with columns ``time_s``, ``acc_x_data``, ``acc_z_data``,
        ``force``. Extra columns are passed through.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = set(DEFAULT_IMU_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(
            f"{csv_path.name} missing expected IMU columns: {sorted(missing)}"
        )
    df["time_s"] = df.index.to_numpy(dtype=float) / float(sample_rate_hz)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_imu_trial(
    df: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Plot ``acc_x``, ``acc_z`` and ``force`` as 3 stacked, time-aligned subplots.

    Args:
        df: Output of :func:`parse_imu_csv`.
        title: Figure title (typically "<MIH> - <trial_key> (<geometry>)").

    Returns:
        Plotly figure with shared x-axis (``time_s``).
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=("acc_x", "acc_z", "force"),
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_x_data"], mode="lines",
                   name="acc_x", line=dict(color="steelblue", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["acc_z_data"], mode="lines",
                   name="acc_z", line=dict(color="seagreen", width=1)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["force"], mode="lines",
                   name="force", line=dict(color="crimson", width=1)),
        row=3, col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="acc_x (m/s^2)", row=1, col=1)
    fig.update_yaxes(title_text="acc_z (m/s^2)", row=2, col=1)
    fig.update_yaxes(title_text="force (N)", row=3, col=1)
    fig.update_layout(
        title=title,
        height=720, width=1100,
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def render_all_trials(
    imu_dir: Path,
    trial_keys: Iterable[str],
    exp_order_for_mih: dict[str, Any] | None = None,
    mih_id: str = "",
    manual_overrides: dict[str, str | Path] | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
) -> dict[str, go.Figure]:
    """Render one figure per trial key.

    Args:
        imu_dir: Output of :func:`find_imu_dir`.
        trial_keys: Ordered trial keys for this MIH.
        exp_order_for_mih: ``experiment_order.json[mih_id]`` block, used to
            stamp each figure title with the geometry. Optional.
        mih_id: Used in figure titles. Optional.
        manual_overrides: Forwarded to :func:`list_trial_imu_files`.
        sample_rate_hz: Forwarded to :func:`parse_imu_csv`.

    Returns:
        Dict ``{trial_key -> plotly Figure}``. Trial keys with no matched
        file are silently skipped.
    """
    files = list_trial_imu_files(imu_dir, trial_keys, manual_overrides=manual_overrides)
    figures: dict[str, go.Figure] = {}
    for key, path in files.items():
        if path is None:
            continue
        try:
            df = parse_imu_csv(path, sample_rate_hz=sample_rate_hz)
        except (KeyError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s for %s: %s", path.name, key, exc)
            continue
        geometry = None
        if exp_order_for_mih is not None:
            geometry = (exp_order_for_mih.get(key) or {}).get("geometry")
        title = f"{mih_id} - {key}"
        if geometry:
            title += f" ({geometry})"
        title += (
            f" - {path.name} - duration={df['time_s'].iloc[-1]:.1f}s "
            f"@ {sample_rate_hz:.0f}Hz"
        )
        figures[key] = plot_imu_trial(df, title=title)
    return figures
