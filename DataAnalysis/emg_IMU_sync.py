import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import dotenv
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.io import savemat, loadmat
from scipy.signal import find_peaks

dotenv.load_dotenv()

DATA_DIR = Path(os.getenv('DATA_DIR')) if os.getenv('DATA_DIR') else None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("emg_imu_sync")

class DataConfig(BaseModel):
    """Runtime configuration for EMG/IMU alignment pipeline.

    Attributes:
        data_dir: Root directory of data. Must contain subject subfolders.
        experiment_order_path: Path to experiment order JSON.
        trial_prefix: Filename prefix used for trials.
        num_trials: Number of trials to process per subject.
        lcimu_fs: Load-cell IMU sampling frequency (Hz).
    """

    data_dir: Path = Field(default_factory=lambda: DATA_DIR if DATA_DIR else Path("."))
    experiment_order_path: Path = Field(
        default_factory=lambda: Path(__file__).parent / "results" / "experiment_order.json"
    )
    trial_prefix: str = "Trial_"
    num_trials: int = 14
    lcimu_fs: int = 500

    def subject_paths(self, subject: str, date: str) -> Dict[str, Path]:
        """Return key directories for a subject/date.

        Args:
            subject: Subject ID (e.g., 'MIH01').
            date: ISO date string.

        Returns:
            Dict with keys 'emg', 'imu', 'opencap', 'root'.
        """
        root = self.data_dir / subject / date
        emg_candidates = [root / "EMG", root / "trials", root / "emg"]
        imu_candidates = [root / "IMU_LoadCell", root / "IMU", root / "imu"]
        opencap_candidates = [root / "OpenCap", root / "Opencap", root / "opencap"]

        def first_existing(paths: Iterable[Path]) -> Path:
            for p in paths:
                if p.exists():
                    return p
            return list(paths)[0]

        return {
            "root": root,
            "emg": first_existing(emg_candidates),
            "imu": first_existing(imu_candidates),
            "opencap": first_existing(opencap_candidates),
        }

    def load_experiment_order(self) -> Dict[str, Dict[str, str]]:
        """Load experiment order mapping from JSON."""
        with open(self.experiment_order_path, "r", encoding="utf-8") as f:
            return json.load(f)


class AlignEMGIMU:
    """Align EMG IMU and Load-cell IMU signals, and detect steps."""

    def __init__(self, emg_dir: Path, imu_dir: Path, lcimu_fs: int) -> None:
        self.emg_dir = emg_dir
        self.imu_dir = imu_dir
        self.lcimu_fs = lcimu_fs

    @staticmethod
    def _first_peak(time_series: np.ndarray, prominence: float, smooth_window: int) -> int:
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            series = np.convolve(time_series, kernel, mode="same")
        else:
            series = time_series
        peaks, _ = find_peaks(series, prominence=prominence)
        if len(peaks) == 0:
            raise ValueError("No peaks found with given prominence")
        return int(peaks[0])

    @staticmethod
    def _load_emg_mat(mat_path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
        mat = loadmat(mat_path)
        try:
            fs_array = np.asarray(mat["Fs"]) 
            time_array = np.asarray(mat["Time"]) 
            data_array = np.asarray(mat["Data"]) 
        except KeyError as exc:
            raise KeyError(f"Missing expected key in {mat_path.name}: {exc}")

        fs = float(np.ravel(fs_array)[9])
        t = np.asarray(time_array)[9, :].astype(float)
        z = np.asarray(data_array)[9, :].astype(float) * -10.0
        return fs, t, z

    def _load_imu_csv(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        t = df["acc_x_time"].to_numpy(dtype=float) / self.lcimu_fs
        z = df["acc_z_data"].to_numpy(dtype=float)
        return t, z

    def align_emg_imu_trial(
        self,
        trial_index: int,
        trial_prefix: str = "Trial_",
        emg_ext: str = ".mat",
        imu_ext: str = ".csv",
        prominence: float = 1.0,
        smooth_window: int = 100,
    ) -> Tuple[float, float]:
        emg_path = self.emg_dir / f"{trial_prefix}{trial_index}{emg_ext}"
        imu_path = self.imu_dir / f"{trial_prefix}{trial_index}{imu_ext}"
        if not emg_path.exists():
            raise FileNotFoundError(f"Missing EMG file: {emg_path}")
        if not imu_path.exists():
            raise FileNotFoundError(f"Missing IMU file: {imu_path}")

        t_lc, z_lc = self._load_imu_csv(imu_path)
        lc_peak_idx = self._first_peak(z_lc, prominence=prominence, smooth_window=smooth_window)
        lc_peak_time = float(t_lc[lc_peak_idx])

        fs_emg, t_emg, z_emg = self._load_emg_mat(emg_path)
        emg_peak_idx = self._first_peak(z_emg, prominence=prominence, smooth_window=smooth_window)
        emg_peak_time = float(t_emg[emg_peak_idx])

        return lc_peak_time, emg_peak_time

    def detect_steps_for_trial(
        self,
        trial_index: int,
        trial_prefix: str = "Trial_",
        imu_ext: str = ".csv",
        min_gap_frames: int = 5,
        window_seconds: float = 0.05,
    ) -> np.ndarray:
        imu_path = self.imu_dir / f"{trial_prefix}{trial_index}{imu_ext}"
        if not imu_path.exists():
            raise FileNotFoundError(f"Missing IMU file: {imu_path}")

        df = pd.read_csv(imu_path)
        force = df["force"].to_numpy(dtype=float)
        _ = df["acc_x_time"].to_numpy(dtype=float)

        window = int(window_seconds * self.lcimu_fs)
        threshold = float(np.nanmean(force))

        starts: List[int] = []
        ends: List[int] = []
        i = 0
        n = len(force)
        while i < n - window:
            if force[i] > threshold and force[i + window] > threshold:
                start_idx = i
                while i < n - window and force[i + window] > threshold:
                    i += 1
                i = min(i + window - 1, n - 1)
                end_idx = i
                if len(starts) == 0 or (start_idx - ends[-1]) >= min_gap_frames:
                    starts.append(start_idx)
                    ends.append(end_idx)
            else:
                i += 1

        starts_seconds = (np.array(starts, dtype=float) * 5.0) / float(self.lcimu_fs)
        return starts_seconds


def process_subject(
    cfg: DataConfig,
    subject: str,
    date: str,
) -> None:
    paths = cfg.subject_paths(subject, date)
    aligner = AlignEMGIMU(paths["emg"], paths["imu"], lcimu_fs=cfg.lcimu_fs)

    peak_indices = np.zeros((cfg.num_trials, 2), dtype=float)
    steps: Dict[str, np.ndarray] = {}

    logger.info("Processing %s %s", subject, date)
    for i in range(1, cfg.num_trials + 1):
        try:
            lc_peak, emg_peak = aligner.align_emg_imu_trial(trial_index=i, trial_prefix=cfg.trial_prefix)
            peak_indices[i - 1, 0] = lc_peak
            peak_indices[i - 1, 1] = emg_peak
        except Exception as exc:
            logger.warning("Trial %d peak alignment failed: %s", i, exc)
            peak_indices[i - 1, :] = np.nan

        try:
            starts_seconds = aligner.detect_steps_for_trial(trial_index=i, trial_prefix=cfg.trial_prefix)
            steps[f"Trial_{i}"] = starts_seconds
        except Exception as exc:
            logger.warning("Trial %d step detection failed: %s", i, exc)
            steps[f"Trial_{i}"] = np.array([], dtype=float)

    sync_path = paths["root"] / "Sync_Indices.xlsx"
    pd.DataFrame(peak_indices, columns=["lcimu_peak_s", "emgimu_peak_s"]).to_excel(sync_path, index=False)
    savemat(paths["root"] / "Step_Indices.mat", {"steps": steps})
    logger.info("Saved sync indices -> %s", sync_path)




if __name__ == "__main__":
    cfg = DataConfig()

    subjects_env = os.getenv("SUBJECTS")
    dates_env = os.getenv("DATES")

    if not subjects_env or not dates_env:
        logger.info("SUBJECTS/DATES not provided; printing example usage and exiting.")
        logger.info(
            "Set SUBJECTS and DATES env vars. Example: SUBJECTS=MIH01 DATES=2025-10-15 python emg_IMU_sync.py"
        )
        logger.info("Experiment order path: %s", cfg.experiment_order_path)
        raise SystemExit(0)

    subjects = [s.strip() for s in subjects_env.split(",") if s.strip()]
    dates = [d.strip() for d in dates_env.split(",") if d.strip()]
    if len(subjects) != len(dates):
        raise SystemExit("SUBJECTS and DATES must have same number of items")

    if not cfg.experiment_order_path.exists():
        logger.warning("Experiment order JSON not found at %s", cfg.experiment_order_path)
    else:
        _ = cfg.load_experiment_order()

    for subj, date in zip(subjects, dates):
        process_subject(cfg, subj, date)