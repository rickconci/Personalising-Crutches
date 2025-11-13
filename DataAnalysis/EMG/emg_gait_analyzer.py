"""
EMG-based Gait Analysis Module

Provides gait analysis capabilities specifically designed for EMG data,
including step detection and gait metrics calculation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from scipy.signal import butter, sosfiltfilt, iirnotch, hilbert, find_peaks
# Removed sliding_window_view to avoid memory issues
from emg_parser import EMGData


class GaitMetrics(BaseModel):
    """Container for gait analysis metrics with Pydantic validation."""
    
    trial_name: str = Field(..., description="Name of the trial")
    step_count: int = Field(..., ge=0, description="Number of detected steps")
    total_duration_s: float = Field(..., gt=0, description="Total trial duration in seconds")
    step_frequency_hz: Optional[float] = Field(None, ge=0, le=10, description="Step frequency in Hz")
    step_regularity_cv: Optional[float] = Field(None, ge=0, le=2, description="Step regularity coefficient of variation")
    step_variance_s2: Optional[float] = Field(None, ge=0, description="Step interval variance in seconds squared")
    mean_step_interval_s: Optional[float] = Field(None, gt=0, description="Mean step interval in seconds")
    min_step_interval_s: Optional[float] = Field(None, gt=0, description="Minimum step interval in seconds")
    max_step_interval_s: Optional[float] = Field(None, gt=0, description="Maximum step interval in seconds")
    mean_emg_amplitude_mv: Optional[float] = Field(None, ge=0, description="Mean EMG amplitude in mV")
    max_emg_amplitude_mv: Optional[float] = Field(None, ge=0, description="Maximum EMG amplitude in mV")
    emg_rms_mv: Optional[float] = Field(None, ge=0, description="EMG RMS value in mV")
    mean_step_emg_mv: Optional[float] = Field(None, ge=0, description="Mean EMG at step times in mV")
    std_step_emg_mv: Optional[float] = Field(None, ge=0, description="Standard deviation of EMG at step times in mV")
    
    @validator('step_frequency_hz', 'step_regularity_cv', 'step_variance_s2', 
               'mean_step_interval_s', 'min_step_interval_s', 'max_step_interval_s',
               'mean_emg_amplitude_mv', 'max_emg_amplitude_mv', 'emg_rms_mv',
               'mean_step_emg_mv', 'std_step_emg_mv')
    def validate_numeric_values(cls, v):
        """Validate numeric values are not NaN or infinite."""
        if v is not None:
            if np.isnan(v) or np.isinf(v):
                return None
        return v
    
    def to_dict(self) -> Dict[str, Union[int, float, str]]:
        """Convert to dictionary for CSV export."""
        return self.dict()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the gait metrics."""
        return {
            'trial_name': self.trial_name,
            'duration_minutes': self.total_duration_s / 60.0,
            'steps_per_minute': (self.step_count / self.total_duration_s) * 60.0 if self.total_duration_s > 0 else 0,
            'step_regularity_rating': self._rate_regularity(),
            'emg_activity_rating': self._rate_emg_activity()
        }
    
    def _rate_regularity(self) -> str:
        """Rate step regularity as excellent, good, fair, or poor."""
        if self.step_regularity_cv is None:
            return "unknown"
        if self.step_regularity_cv < 0.1:
            return "excellent"
        elif self.step_regularity_cv < 0.2:
            return "good"
        elif self.step_regularity_cv < 0.3:
            return "fair"
        else:
            return "poor"
    
    def _rate_emg_activity(self) -> str:
        """Rate EMG activity level."""
        if self.mean_emg_amplitude_mv is None:
            return "unknown"
        if self.mean_emg_amplitude_mv < 0.1:
            return "low"
        elif self.mean_emg_amplitude_mv < 0.5:
            return "moderate"
        else:
            return "high"


class EMGGaitAnalyzer:
    """
    Gait analysis for EMG-based step detection.
    
    Provides step detection and gait metrics calculation specifically
    designed for EMG data from forearm muscles.
    """
    
    def __init__(self, sampling_frequency: float = 2000.0):
        """
        Initialize the EMG gait analyzer.
        
        Args:
            sampling_frequency: Sampling frequency in Hz
        """
        self.fs = sampling_frequency
    
    def detect_steps_from_emg_robust(
        self,
        emg_data,
        muscle: str = "forearm",
        notch_freq: float = 50.0,           # 60.0 in US mains
        bandpass_range: Tuple[float, float] = (20.0, 250.0),
        envelope_method: str = "hilbert",   # "hilbert" or "lpf"
        envelope_cutoff: float = 1.0,       # used if envelope_method=="lpf"
        highpass_env_hz: float = 0.3,       # removes baseline drift on envelope
        min_step_distance_s: float = 1.3,   # enforce ~1.5–2 s cadence
        prominence_z: float = 2.0,
        min_peak_width_s: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Fast, robust EMG step detection using scipy-based signal processing.
        
        Implements the complete pipeline:
        1. Clean signal (notch + bandpass)
        2. Extract envelope (Hilbert or LPF)
        3. Remove baseline drift (high-pass)
        4. Robust normalization (median/MAD)
        5. Peak picking with refractory period
        
        Args:
            emg_data: Parsed EMG data
            muscle: Muscle to analyze (default: 'forearm')
            notch_freq: Notch filter frequency (50 or 60 Hz)
            bandpass_range: Bandpass filter range (low, high) in Hz
            envelope_method: "hilbert" or "lpf" for envelope extraction
            envelope_cutoff: Low-pass cutoff for envelope in Hz (if method="lpf")
            highpass_env_hz: High-pass cutoff to remove baseline drift
            min_step_distance_s: Minimum distance between steps in seconds
            prominence_z: Peak prominence threshold in z-units
            min_peak_width_s: Minimum peak width in seconds
            
        Returns:
            Tuple of (step_times, step_indices, processing_info)
        """
        muscle_col = f"{muscle}_mV"
        if muscle_col not in emg_data.data.columns:
            raise ValueError(f"Muscle {muscle} not found")

        t = emg_data.data["time_s"].to_numpy()
        x = emg_data.data[muscle_col].to_numpy()
        fs = float(getattr(emg_data, "sampling_rate", self.fs))
        self.fs = fs  # keep in sync

        # 1) Clean (notch + bandpass)
        x = self._clean_emg_scipy(x, notch_freq, bandpass_range)

        # 2) Envelope
        env = self._envelope(x, method=envelope_method, lpf_cutoff=envelope_cutoff)

        # 3) Remove slow drift on envelope (high-pass ~0.3 Hz) and robust-normalize
        env_hp = self._butter_highpass(env, highpass_env_hz)
        z = self._robust_z(env_hp)  # global median/MAD (fast, vectorized)

        # 4) Adaptive peak picking on envelope
        step_times, peaks, adaptive_info = self._detect_adaptive_peaks(
            env, t, fs,
            thr_win_s=15.0, thr_pct=85,
            base_distance_s=min_step_distance_s,
            min_width_s=min_peak_width_s,
            abs_floor_mv=0.10
        )

        # Memory-efficient processing info
        decim = adaptive_info["decim"]
        info = {
            "time_s": t[::decim],  # downsampled time for debug plots
            "envelope_ds": adaptive_info.get("envelope_ds", env[::decim]),  # downsampled envelope
            "thr_ds": adaptive_info["thr_ds"],  # downsampled threshold
            "decim": decim,
            "cleaned_emg_sample": x[::10],  # light preview to save memory
            "bandpass_range": bandpass_range,
            "notch_freq": notch_freq,
            "envelope_method": envelope_method,
            "envelope_cutoff": envelope_cutoff,
            "adaptive_detection": True
        }
        return step_times, peaks, info
    
    def analyze_gait(
        self, 
        emg_data: EMGData, 
        step_times: np.ndarray,
        trial_name: str,
        muscle: str = 'forearm'
    ) -> GaitMetrics:
        """
        Calculate comprehensive gait metrics from EMG data and step times.
        
        Args:
            emg_data: Parsed EMG data
            step_times: Detected step times
            trial_name: Name of the trial
            muscle: Muscle analyzed
            
        Returns:
            GaitMetrics object with calculated metrics
        """
        muscle_col = f"{muscle}_mV"
        
        # Basic metrics
        total_duration = float(emg_data.data['time_s'].max() - emg_data.data['time_s'].min())
        step_count = len(step_times)
        
        metrics = GaitMetrics(
            trial_name=trial_name,
            step_count=step_count,
            total_duration_s=total_duration
        )
        
        if step_count >= 2:
            # Calculate step intervals
            step_intervals = np.diff(step_times)
            
            # Step timing metrics
            metrics.mean_step_interval_s = float(np.mean(step_intervals))
            metrics.step_frequency_hz = float(1.0 / np.mean(step_intervals))
            metrics.step_regularity_cv = float(np.std(step_intervals) / np.mean(step_intervals))
            metrics.step_variance_s2 = float(np.var(step_intervals))
            metrics.min_step_interval_s = float(np.min(step_intervals))
            metrics.max_step_interval_s = float(np.max(step_intervals))
        
        # EMG-based metrics
        if muscle_col in emg_data.data.columns:
            emg_signal = emg_data.data[muscle_col].values
            abs_emg = np.abs(emg_signal)
            
            metrics.mean_emg_amplitude_mv = float(np.mean(abs_emg))
            metrics.max_emg_amplitude_mv = float(np.max(abs_emg))
            metrics.emg_rms_mv = float(np.sqrt(np.mean(emg_signal**2)))
            
            # EMG at step times
            if len(step_times) > 0:
                step_emg_values = np.interp(step_times, emg_data.data['time_s'], abs_emg)
                metrics.mean_step_emg_mv = float(np.mean(step_emg_values))
                metrics.std_step_emg_mv = float(np.std(step_emg_values))
        
        return metrics
    
    def filter_trial_by_duration(
        self, 
        emg_data: EMGData, 
        min_duration: float = 60.0, 
        max_duration: float = 120.0
    ) -> Optional[EMGData]:
        """
        Filter trial by duration requirements.
        
        Args:
            emg_data: EMG data to filter
            min_duration: Minimum trial duration in seconds
            max_duration: Maximum trial duration in seconds
            
        Returns:
            Filtered EMGData or None if trial doesn't meet criteria
        """
        total_duration = emg_data.data['time_s'].max() - emg_data.data['time_s'].min()
        
        if total_duration < min_duration:
            return None
        
        # Crop to max_duration if needed
        if total_duration > max_duration:
            start_time = emg_data.data['time_s'].min()
            end_time = start_time + max_duration
            filtered_data = emg_data.data[
                (emg_data.data['time_s'] >= start_time) & 
                (emg_data.data['time_s'] <= end_time)
            ].copy()
            
            # Create new EMGData object with filtered data
            return EMGData(
                data=filtered_data,
                metadata=emg_data.metadata,
                roster=emg_data.roster,
                sampling_rate=emg_data.sampling_rate
            )
        
        return emg_data
    
    def _moving_average_smooth(self, signal: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Apply aggressive moving average smoothing.
        
        Args:
            signal: Input signal
            k: Points on each side for moving average
            
        Returns:
            Smoothed signal
        """
        if len(signal) < 2 * k + 1:
            return signal
        
        smoothed = np.zeros_like(signal)
        
        for i in range(len(signal)):
            start_idx = max(0, i - k)
            end_idx = min(len(signal), i + k + 1)
            smoothed[i] = np.mean(signal[start_idx:end_idx])
        
        return smoothed
    
    def _find_peaks_simple(
        self, 
        signal: np.ndarray, 
        height: float, 
        distance: int
    ) -> np.ndarray:
        """Simple peak detection using numpy operations."""
        peaks = []
        for i in range(1, len(signal) - 1):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                signal[i] > height):
                peaks.append(i)
        
        peaks = np.array(peaks)
        
        # Filter by distance
        if len(peaks) > 1:
            filtered_peaks = [peaks[0]]
            for peak in peaks[1:]:
                if peak - filtered_peaks[-1] >= distance:
                    filtered_peaks.append(peak)
            peaks = np.array(filtered_peaks)
        
        return peaks
    
    def _clean_emg_scipy(self, x, notch_freq, bp):
        """Clean EMG signal using scipy filters (notch + bandpass)."""
        x = x - np.mean(x)  # remove DC
        # Notch fundamental (and its 2nd harmonic helps in noisy rooms)
        if notch_freq > 0:
            for f0 in (notch_freq, 2*notch_freq):
                b, a = iirnotch(w0=f0, Q=30, fs=self.fs)
                x = sosfiltfilt(self._ba2sos(b, a), x)
        # Band-pass (Butterworth, SOS for stability)
        lo, hi = bp
        sos = butter(4, [lo, hi], btype="band", fs=self.fs, output="sos")
        return sosfiltfilt(sos, x)
    
    def _envelope(self, x, method="hilbert", lpf_cutoff=1.0):
        """Extract EMG envelope using Hilbert transform or low-pass filtering."""
        rect = np.abs(x)
        if method == "hilbert":
            # Analytic signal → true amplitude envelope; already smooth-ish
            env = np.abs(hilbert(x))
            # Optional light LPF to tame ripple
            sos = butter(2, 2.0, btype="low", fs=self.fs, output="sos")
            return sosfiltfilt(sos, env)
        else:
            # Rectify + low-pass (classic EMG envelope)
            sos = butter(2, lpf_cutoff, btype="low", fs=self.fs, output="sos")
            return sosfiltfilt(sos, rect)
    
    def _butter_highpass(self, x, cutoff):
        """High-pass filter using Butterworth filter."""
        sos = butter(2, cutoff, btype="high", fs=self.fs, output="sos")
        return sosfiltfilt(sos, x)

    @staticmethod
    def _robust_z(x):
        """Robust z-score normalization using median and MAD."""
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-12
        return (x - med) / (1.4826 * mad)

    @staticmethod
    def _ba2sos(b, a):
        """Convert (b,a) to SOS for sosfiltfilt."""
        from scipy.signal import tf2sos
        return tf2sos(b, a)
    
    def _lp(self, x, fs, fc, order=2):
        """Low-pass filter using Butterworth."""
        sos = butter(order, fc, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, x)
    
    def _rolling_percentile_safe(self, x, fs, win_s=15.0, p=85):
        """Memory-safe rolling percentile using pandas."""
        w = max(3, int(win_s * fs))
        s = pd.Series(x)
        return s.rolling(window=w, center=True, min_periods=1).quantile(p/100.0).to_numpy()
    
    def _local_period_from_acf(self, env, fs, min_s=1.0, max_s=2.5):
        """Estimate local period from autocorrelation of envelope segment."""
        n = len(env)
        # zero-mean
        y = env - np.mean(env)
        acf = np.correlate(y, y, mode='full')[n-1:]
        lmin = int(min_s*fs)
        lmax = int(max_s*fs)
        lag = lmin + np.argmax(acf[lmin:lmax])
        return max(lmin, min(lmax, lag))
    
    def _detect_adaptive_peaks(
        self, env, t, fs,
        thr_win_s=15.0, thr_pct=85,
        base_distance_s=1.2,
        min_width_s=0.25,
        target_env_fs=20.0,
        abs_floor_mv=0.10           # absolute envelope floor
    ):
        """
        Enhanced adaptive peak detection with tail-safe rolling percentiles and tail repair.
        
        Args:
            env: EMG envelope signal
            t: Time vector
            fs: Sampling frequency
            thr_win_s: Window size for rolling threshold in seconds
            thr_pct: Percentile for rolling threshold (80-90)
            base_distance_s: Base distance between peaks in seconds
            min_width_s: Minimum peak width in seconds
            target_env_fs: Target sampling rate for processing (20 Hz)
            abs_floor_mv: Absolute envelope floor in mV
            
        Returns:
            Tuple of (step_times, step_indices, extra_info)
        """
        # 1) Smooth + decimate
        sos = butter(2, 0.9, btype="low", fs=fs, output="sos")
        env_s = sosfiltfilt(sos, env)
        decim = max(1, int(round(fs / target_env_fs)))
        env_ds = env_s[::decim]
        t_ds = t[::decim]
        fs_ds = fs / decim

        # 2) TAIL-SAFE rolling percentile
        w = max(3, int(thr_win_s * fs_ds))
        s = pd.Series(env_ds)
        thr_center = s.rolling(window=w, center=True, min_periods=max(1, w//4)).quantile(thr_pct/100.0)
        thr_trail = s.rolling(window=w, center=False, min_periods=1).quantile(thr_pct/100.0)
        thr = np.where(np.isfinite(thr_center.values), thr_center.values, thr_trail.values)

        # 3) Hybrid height: adaptive OR absolute floor
        height = np.maximum(thr, abs_floor_mv)

        dist0 = int(base_distance_s * fs_ds)
        width0 = int(min_width_s * fs_ds)

        # 4) First pass
        peaks_ds, props = find_peaks(env_ds, height=height, distance=dist0, width=width0, rel_height=0.75)

        # 5) Adaptive distance from local tempo
        if peaks_ds.size >= 4:
            i0, i1 = max(peaks_ds[0]-int(10*fs_ds), 0), min(peaks_ds[-1]+int(10*fs_ds), len(env_ds))
            y = env_ds[i0:i1] - np.mean(env_ds[i0:i1])
            acf = np.correlate(y, y, mode='full')[len(y)-1:]
            lmin = int(1.0*fs_ds)
            lmax = int(2.5*fs_ds)
            per = lmin + int(np.argmax(acf[lmin:lmax]))
            dist = max(dist0, int(0.6 * per))
            peaks_ds, props = find_peaks(env_ds, height=height, distance=dist, width=width0, rel_height=0.75)

        # 6) GAP REPAIR + TAIL REPAIR
        add = []

        if peaks_ds.size >= 2:
            intervals = np.diff(peaks_ds)
            med = np.median(intervals)
            miss_idx = np.where(intervals > 2.2 * med)[0]
            for k in miss_idx:
                a, b = peaks_ds[k], peaks_ds[k+1]
                seg = slice(a + int(0.15*med), b - int(0.15*med))
                if seg.stop - seg.start > 5:
                    loc_thr = max(abs_floor_mv, np.percentile(env_ds[seg], 70))
                    loc_dist = int(0.5 * med)
                    cand, _ = find_peaks(env_ds[seg], height=loc_thr, distance=loc_dist, width=width0, rel_height=0.75)
                    if cand.size:
                        add.append(seg.start + cand[np.argmax(env_ds[seg][cand])])

        # ---- TAIL REPAIR ----
        # look at the last ~8–10 s and allow slightly closer spacing
        tail_len = int(10 * fs_ds)
        tail = slice(max(0, len(env_ds) - tail_len), len(env_ds))
        if peaks_ds.size:
            last = peaks_ds[-1]
            tail = slice(max(last + int(0.3*(np.median(np.diff(peaks_ds)) if peaks_ds.size>1 else fs_ds)),
                             len(env_ds) - tail_len),
                         len(env_ds))
        if tail.stop - tail.start > 5:
            tail_thr = max(abs_floor_mv, np.percentile(env_ds[tail], 70))
            # let the last couple be a bit closer (0.6× recent median)
            loc_dist_tail = int(0.6 * (np.median(np.diff(peaks_ds[-5:])) if peaks_ds.size>5 else dist0))
            cand, _ = find_peaks(env_ds[tail], height=tail_thr, distance=loc_dist_tail, width=width0, rel_height=0.75)
            if cand.size:
                add.extend(tail.start + cand)

        if add:
            peaks_ds = np.sort(np.unique(np.r_[peaks_ds, np.array(add, dtype=int)]))

        # 7) Map back to full-rate indices
        peaks = peaks_ds * decim
        step_times = t[peaks]
        return step_times, peaks, {
            "peak_heights_ds": props.get("peak_heights", None),
            "thr_ds": thr,
            "decim": decim,
            "envelope_ds": env_ds
        }
    
