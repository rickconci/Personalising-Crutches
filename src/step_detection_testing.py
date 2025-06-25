import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt, find_peaks_cwt
from typing import List, Dict
import json, pathlib, plotly.graph_objs as go
from scipy.signal import correlate, lfilter
from dl_step_detection import TimeSFMStepDetector


def butter_bandpass(x, fs, low, high, order=4):
    b, a = butter(order, [low, high], btype='bandpass', fs=fs)
    return filtfilt(b, a, x)

def match_events(pred: np.ndarray, truth: np.ndarray, tol: float = 0.25) -> np.ndarray:
    """
    Return a boolean array `matches` same length as pred; True if pred[i]
    is within `tol` seconds of any truth sample (one‑to‑one greedy match).
    """
    matches = np.zeros_like(pred, dtype=bool)
    used = np.zeros_like(truth, dtype=bool)
    for i, t in enumerate(pred):
        idx = np.argmin(np.abs(truth - t))
        if not used[idx] and np.abs(truth[idx] - t) <= tol:
            matches[i] = True
            used[idx] = True
    return matches


class StepDetectionTesting:
    def __init__(self, input_path, ground_truth_path, smooth_fraction=25):
        self.luke_test_df = pd.read_parquet(input_path)
        self.ground_truth_steps_df = pd.read_csv(ground_truth_path)
        self.luke_test_df['time'] =(self.luke_test_df['acc_x_time'] - self.luke_test_df['acc_x_time'].iloc[0])/1000
        # estimate sampling frequency from time column if present; default 100 Hz
        if 'time' in self.luke_test_df.columns and pd.to_numeric(self.luke_test_df['time'], errors='coerce').notna().all():
            times = self.luke_test_df['time'].dropna().values
            if len(times) > 1:
                self.fs = 1.0 / np.median(np.diff(times))
            else:
                self.fs = 100.0
        else:
            self.fs = 100.0

        # Preprocessing: exponential smoothing + vector magnitude
        alpha = smooth_fraction / 100.0
        acc_x_smooth = lfilter([alpha], [1, -(1 - alpha)], self.luke_test_df['acc_x_data'])
        acc_z_smooth = lfilter([alpha], [1, -(1 - alpha)], self.luke_test_df['acc_z_data'])
        self.processed_signal = np.sqrt(acc_x_smooth**2 + acc_z_smooth**2)
        
        # Prepare multi-channel signal for deep learning models
        signals_to_stack = [self.processed_signal]
        self.force = None
        self.force_gradient_signal = None
        if 'force' in self.luke_test_df.columns:
            self.force = self.luke_test_df['force'].values
            signals_to_stack.append(self.force)
            # Compute the time derivative (gradient) of the force signal
            time = self.luke_test_df['time'].values
            self.force_gradient_signal = np.gradient(self.force, time)

        self.multichannel_signal = np.stack(signals_to_stack, axis=1)

    def plot_processed_signal(self, peaks: List[float] = None):
        plt.figure(figsize=(10, 3))
        t = np.arange(len(self.processed_signal)) / self.fs
        ground_truth_times = self.ground_truth_steps_df['Step Times'].values

        # --- Matplotlib plot (for static PNGs) ---
        plt.plot(t, self.processed_signal, 'g', lw=0.8, label='Processed Accel')
        plt.plot(ground_truth_times, np.interp(ground_truth_times, t, self.processed_signal), 'r^', markersize=8, label='Ground Truth')
        if peaks is not None:
            plt.plot(peaks, np.interp(peaks, t, self.processed_signal), 'bo', label='Detected Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Smoothed Vector Magnitude')
        plt.legend()
        plt.tight_layout()

        # --- Plotly figure (for interactive HTML) ---
        fig_plotly = go.Figure()

        # 1. Processed Accelerometer Signal
        fig_plotly.add_trace(go.Scatter(x=t, y=self.processed_signal,
                                        mode='lines', name='Processed Accel', yaxis='y1'))

        # 2. Ground Truth Steps
        fig_plotly.add_trace(go.Scatter(x=ground_truth_times,
                                        y=np.interp(ground_truth_times, t, self.processed_signal),
                                        mode='markers', marker=dict(color='red', symbol='cross', size=8),
                                        name='Ground Truth'))

        # 3. Detected Peaks
        if peaks is not None:
            fig_plotly.add_trace(go.Scatter(x=peaks,
                                            y=np.interp(peaks, t, self.processed_signal),
                                            mode='markers', marker=dict(color='blue', size=6),
                                            name='Detected Peaks'))

        # 4. Force signal on secondary axis, if available
        layout_update = {
            'title_text': 'Step Detection Analysis',
            'xaxis_title': 'Time (s)',
            'yaxis_title': 'Smoothed Vector Magnitude',
            'legend_title_text': 'Signals'
        }
        if self.force is not None:
            fig_plotly.add_trace(go.Scatter(x=t, y=self.force,
                                            mode='lines', name='Force',
                                            line=dict(dash='dot'),
                                            yaxis='y2'))
            layout_update['yaxis2'] = {
                'title': 'Force',
                'overlaying': 'y',
                'side': 'right'
            }
        
        fig_plotly.update_layout(**layout_update)

        self._last_plotly_fig = fig_plotly  # stash for saving
        #plt.show()

    def step_detection_algo_1(self, use_force_gradient=False):
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 1 with force gradient requested but not available. Skipping.")
                return np.array([])
            signal = self.force_gradient_signal
        else:
            signal = self.processed_signal
        # Savitzky‑Golay smoothing then prominence‑based peak finding
        sig = savgol_filter(signal, 15, 3)
        peaks, _ = find_peaks(sig, prominence=np.std(sig)*1.0, distance=self.fs*0.2)
        return peaks / self.fs  # return seconds

    def step_detection_algo_2(self, use_force_gradient=False):
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 2 with force gradient requested but not available. Skipping.")
                return np.array([])
            signal = self.force_gradient_signal
        else:
            signal = self.processed_signal
        sig = butter_bandpass(signal, self.fs, 1, 10)
        widths = np.arange(int(0.02*self.fs), int(0.06*self.fs))
        peaks = find_peaks_cwt(sig, widths)
        return np.array(peaks) / self.fs

    def step_detection_algo_3(self, use_force_gradient=False):
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 3 with force gradient requested but not available. Skipping.")
                return np.array([])
            signal = self.force_gradient_signal
        else:
            signal = self.processed_signal
        sig = signal
        deriv = np.diff(sig)
        zero_cross = np.where((deriv[:-1] > 0) & (deriv[1:] < 0))[0] + 1
        # amplitude filter
        threshold = np.median(sig) + 0.8 * np.std(sig)
        peaks = zero_cross[sig[zero_cross] > threshold]
        # enforce minimum distance of 0.2 s
        peaks = peaks[np.insert(np.diff(peaks) > 0.2*self.fs, 0, True)]
        return peaks / self.fs

    # ---------------------------------------------------------------------
# NEW -- Algorithm 4: TKEO + peaks
# ---------------------------------------------------------------------
    def step_detection_algo_4(self, use_force_gradient=False):
        """
        Teager-Kaiser Energy Operator emphasises impulsive events.
        1) compute TKEO     E[n] = x[n]**2 - x[n-1]*x[n+1]
        2) smooth with Sav-Gol
        3) find prominences
        """
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 4 with force gradient requested but not available. Skipping.")
                return np.array([])
            x = self.force_gradient_signal
        else:
            x = self.processed_signal
        # pad so len matches
        tkeo = np.empty_like(x)
        tkeo[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
        tkeo[0] = tkeo[-1] = 0
        tkeo_s = savgol_filter(tkeo, 15, 3)
        prom = np.median(np.abs(tkeo_s)) * 5        # 5× MAD
        peaks, _ = find_peaks(tkeo_s,
                              prominence=prom,
                              distance=0.3 * self.fs)
        return peaks / self.fs

# ---------------------------------------------------------------------
# NEW -- Algorithm 5: Matched-filter correlation
# ---------------------------------------------------------------------
    def step_detection_algo_5(self, template_len_s: float = 0.6, use_force_gradient=False):
        """
        Builds a template from the first few ground-truth steps, then
        correlates the entire signal with that template and picks peaks
        on the correlation trace.
        """
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 5 with force gradient requested but not available. Skipping.")
                return np.array([])
            signal = self.force_gradient_signal
        else:
            signal = self.processed_signal

        truth = self.ground_truth_steps_df['Step Times'].values
        fs = self.fs
        L = int(template_len_s * fs)

        # ---- build template from first 3 labelled steps
        segments = []
        for t0 in truth[:3]:
            i0 = int(t0 * fs)
            if i0 - L//2 >= 0 and i0 + L//2 < len(signal):
                seg = signal[i0-L//2:i0+L//2]
                segments.append(seg)
        if not segments:
            return np.empty(0)
        template = np.mean(segments, axis=0)
        template = (template - template.mean()) / (template.std() + 1e-6)

        # ---- correlate & peak-pick
        sig = signal
        sig_n = (sig - sig.mean()) / (sig.std() + 1e-6)
        corr = correlate(sig_n, template, mode='same')
        prom = np.median(np.abs(corr)) * 5
        peaks, _ = find_peaks(corr,
                              prominence=prom,
                              distance=0.3 * fs)
        return peaks / fs

# ---------------------------------------------------------------------
# NEW -- Algorithm 6: From Accelerometer_Processing_Program.html
# ---------------------------------------------------------------------
    def step_detection_algo_6(self, use_force_gradient=False):
        """
        Port of the step detection logic from Accelerometer_Processing_Program.html
        Now uses the globally pre-processed signal and an adaptive threshold.
        """
        if use_force_gradient:
            if self.force_gradient_signal is None:
                print("Warning: Algo 6 with force gradient requested but not available. Skipping.")
                return np.array([])
            sig = self.force_gradient_signal
        else:
            sig = self.processed_signal

        # --- Step detection logic
        # 1. Calculate derivative
        differential_duration = 0.24  # seconds
        differential_points = int(round(differential_duration * self.fs))
        deriv = np.zeros_like(sig)
        if differential_points > 0:
            time_interval = differential_points / self.fs
            deriv[differential_points:] = (sig[differential_points:] - sig[:-differential_points]) / time_interval

        # 2. Adaptive threshold based on derivative signal's median absolute deviation
        threshold = np.median(np.abs(deriv)) * 5.0

        # 3. Find upward crossings of the threshold
        crossings = np.where((deriv[:-1] < threshold) & (deriv[1:] >= threshold))[0] + 1

        # 4. Refine step start by finding peak in subsequent window
        peak_indices = []
        for i in crossings:
            window_end = min(i + 100, len(sig))
            if window_end > i:
                max_in_window_idx = np.argmax(sig[i:window_end])
                peak_indices.append(i + max_in_window_idx)
        
        if not peak_indices:
            return np.array([])
            
        # 5. Filter out peaks that are too close (10 samples in JS)
        unique_peaks = []
        last_peak = -np.inf
        for peak_idx in peak_indices:
            if peak_idx - last_peak > 10:
                unique_peaks.append(peak_idx)
                last_peak = peak_idx
        
        return np.array(unique_peaks) / self.fs

    def step_detection_algo_7_force_deriv(self):
        """
        Detects steps based on the derivative of the force signal.
        A step is detected when there is a sharp decrease in force (trough in derivative),
        indicating the person has landed after vaulting.
        The detection threshold is set dynamically based on the average trough depth.
        """
        if 'force' not in self.luke_test_df.columns:
            print("Warning: 'force' column not found for algo_7_force_deriv. Skipping.")
            return np.array([])

        force = self.luke_test_df['force'].values
        time = self.luke_test_df['time'].values

        # Compute the time derivative (gradient) of the force signal
        d_force_dt = np.gradient(force, time)

        # 1. Identify initial troughs in the derivative using prominence to avoid noise
        initial_prominence = np.std(d_force_dt) * 0.2
        initial_troughs, _ = find_peaks(-d_force_dt, prominence=initial_prominence)

        if len(initial_troughs) < 5:
            print("Warning: Not enough distinct troughs in force derivative; using fallback threshold.")
            height_threshold = np.std(d_force_dt) * 2.0
        else:
            # 2. Compute average minimum values (depths of troughs)
            avg_trough_depth = np.mean(-d_force_dt[initial_troughs])
            height_threshold = 0.9 * avg_trough_depth

        # 3. Final detection with a refractory period (min 200ms between detections)
        min_dist_samples = int(0.2 * self.fs)
        final_troughs, _ = find_peaks(-d_force_dt, height=height_threshold, distance=min_dist_samples)

        return final_troughs / self.fs  # Return timestamps in seconds

    def save_summary_report(self, sorted_metrics: List[Dict], outdir: str):
        """
        Saves a summary report of all algorithm results, ranked by F1 score.
        """
        outdir = pathlib.Path(outdir)
        report_path = outdir / "summary_report.md"
        
        with open(report_path, "w") as f:
            f.write("# Step Detection Algorithm Performance Summary\n\n")
            f.write("Algorithms are ranked by their F1 score in descending order.\n\n")
            
            # Create a markdown table header
            headers = ["Rank", "Algorithm", "F1", "Precision", "Recall", "TP", "FP", "FN", "Accuracy", "ISV Error", "SF Error"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "---|" * len(headers) + "\n")
            
            # Populate the table
            for i, metrics in enumerate(sorted_metrics):
                rank = i + 1
                row = [
                    str(rank),
                    metrics.get('algorithm', 'N/A'),
                    f"{metrics.get('f1', 0):.4f}",
                    f"{metrics.get('precision', 0):.4f}",
                    f"{metrics.get('recall', 0):.4f}",
                    str(metrics.get('TP', 0)),
                    str(metrics.get('FP', 0)),
                    str(metrics.get('FN', 0)),
                    f"{metrics.get('accuracy', 0):.4f}",
                    f"{metrics.get('isv_error', 0):.4f}",
                    f"{metrics.get('sf_error', 0):.4f}",
                ]
                f.write("| " + " | ".join(row) + " |\n")
                
        print(f"\nSaved summary report → {report_path}")

    @staticmethod
    def _postprocess_steps(preds: np.ndarray, tolerance_ratio: float = 0.2, isolation_threshold: float = 5.0) -> np.ndarray:
        """
        Applies post-processing filters to a series of detected step times.
        1. Regularizes step intervals based on a data-driven expected interval.
        2. Removes isolated steps that are too far from any other step.
        """
        if len(preds) < 2:
            return preds

        # 1. Determine the data-driven expected interval from the median of diffs
        intervals = np.diff(preds)
        # Use median for robustness; fallback if there are too few intervals to be reliable.
        expected_interval = np.median(intervals) if len(intervals) >= 3 else 1.1

        # --- Interval-based regularization ---
        min_conflict_interval = expected_interval * tolerance_ratio
        
        regularized_preds = [preds[0]]
        for i in range(1, len(preds)):
            current_pred = preds[i]
            last_accepted_pred = regularized_preds[-1]
            
            if current_pred - last_accepted_pred < min_conflict_interval:
                # Conflict: this point is too close to the previous one.
                # Decide whether to keep the last point or swap it for the current one
                # based on which one fits the expected interval trend better.
                prev_accepted_pred = regularized_preds[-2] if len(regularized_preds) > 1 else 0.0

                error_if_keep_last = abs((last_accepted_pred - prev_accepted_pred) - expected_interval)
                error_if_use_current = abs((current_pred - prev_accepted_pred) - expected_interval)

                if error_if_use_current < error_if_keep_last:
                    regularized_preds[-1] = current_pred  # Swap
            else:
                regularized_preds.append(current_pred)
        
        regularized_preds = np.array(regularized_preds)

        # --- Isolation filter ---
        if len(regularized_preds) < 2:
            # If only one step remains, it is by definition isolated, so remove it.
            # If there are no steps, return empty.
            return np.array([])
            
        final_preds = []
        # Pad with infinities to handle endpoints gracefully. A point must be far from BOTH neighbors to be isolated.
        padded_preds = np.concatenate(([np.NINF], regularized_preds, [np.PINF]))
        
        for i in range(1, len(padded_preds) - 1):
            dist_to_prev = padded_preds[i] - padded_preds[i-1]
            dist_to_next = padded_preds[i+1] - padded_preds[i]
            
            is_isolated = (dist_to_prev > isolation_threshold) and (dist_to_next > isolation_threshold)
                          
            if not is_isolated:
                final_preds.append(padded_preds[i])
                
        return np.array(final_preds)

    def train_timesfm(self, model_name="google/timesfm-2.0-500m-pytorch",
                      output_dir="finetuned_step_fm",
                      win_sec=2.0, stride_sec=1.0,
                      epochs=10, lr=2e-4, batch_size=32,
                      freeze_layers=0):
        detector = TimeSFMStepDetector(model_name=model_name)
        detector.train(
            multichannel_signal=self.multichannel_signal,
            times=self.luke_test_df['time'].values,
            gt_steps=self.ground_truth_steps_df['Step Times'].values,
            fs=self.fs,
            output_dir=output_dir,
            win_sec=win_sec,
            stride_sec=stride_sec,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            freeze_layers=freeze_layers
        )

    def step_detection_deep(self, ckpt="finetuned_step_fm",
                            win_sec=2.0, stride_sec=0.25, threshold=0.9):
        detector = TimeSFMStepDetector()
        return detector.predict(
            multichannel_signal=self.multichannel_signal,
            processed_signal_for_refinement=self.processed_signal,
            times=self.luke_test_df['time'].to_numpy(),
            fs=self.fs,
            ckpt=ckpt,
            win_sec=win_sec,
            stride_sec=stride_sec,
            threshold=threshold
        )

    def evaluate_step_detection(self, preds: np.ndarray) -> Dict[str, float]:
        truth = self.ground_truth_steps_df['Step Times'].values
        matches = match_events(preds, truth, tol=0.25)

        tp = matches.sum()
        fp = len(preds) - tp
        fn = len(truth) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0
        accuracy  = tp / max(len(truth), len(preds))

        # inter‑step variability (ISV)
        gt_intervals = np.diff(truth)
        pr_intervals = np.diff(preds)
        isv_gt = np.std(gt_intervals)
        isv_pr = np.std(pr_intervals)
        isv_error = abs(isv_pr - isv_gt)

        # step frequency (Hz)
        sf_gt = 1.0 / np.mean(gt_intervals)
        sf_pr = 1.0 / np.mean(pr_intervals)
        sf_error = abs(sf_pr - sf_gt)

        return {
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy,
            "isv_gt": isv_gt, "isv_pred": isv_pr, "isv_error": isv_error,
            "sf_gt": sf_gt, "sf_pred": sf_pr, "sf_error": sf_error,
        }

    def save_artifacts(self, outdir: str, algo_name: str,
                       metrics: Dict[str, float], peaks: np.ndarray):
        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # save metrics json
        metrics_path = outdir / f"{algo_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics → {metrics_path}")

        # save static PNG
        png_path = outdir / f"{algo_name}_plot.png"
        plt.savefig(png_path, dpi=200)
        print(f"Saved Matplotlib plot → {png_path}")

        # save plotly HTML
        html_path = outdir / f"{algo_name}_plotly.html"
        self._last_plotly_fig.write_html(html_path)
        print(f"Saved interactive Plotly plot → {html_path}")


if __name__ == "__main__":
    input_path = '/Users/riccardoconci/Library/Mobile Documents/com~apple~CloudDocs/HQ_2024/Projects/2024_Harvard_AIM/Research/OPMO/Personalising-Crutches/2025.06.18/Luke_test2.parquet'
    ground_truth_path = '/Users/riccardoconci/Library/Mobile Documents/com~apple~CloudDocs/HQ_2024/Projects/2024_Harvard_AIM/Research/OPMO/Personalising-Crutches/2025.06.18/steps_ground_truth_LukeTest.csv'

    tester = StepDetectionTesting(input_path, ground_truth_path)

    # --- Run and evaluate algorithms ---
    unfiltered_algos = {}
    base_algos = {
        "algo1": tester.step_detection_algo_1,
        "algo2": tester.step_detection_algo_2,
        "algo3": tester.step_detection_algo_3,
        "algo4_TKEO": tester.step_detection_algo_4,
        "algo5_Matched": tester.step_detection_algo_5,
        "algo6_JS": tester.step_detection_algo_6,
    }

    for name, func in base_algos.items():
        # Run on default (accelerometer) signal
        unfiltered_algos[f"{name}_accel"] = func(use_force_gradient=False)
        # Run on force gradient signal
        unfiltered_algos[f"{name}_force_grad"] = func(use_force_gradient=True)

    # Add other specific algorithms
    unfiltered_algos["algo7_Force_Deriv"] = tester.step_detection_algo_7_force_deriv()
    unfiltered_algos["deep_learning"] = tester.step_detection_deep(ckpt="finetuned_step_fm")

    # Apply post-processing filters to all algorithms
    algos = {name: tester._postprocess_steps(pred) for name, pred in unfiltered_algos.items()}
    
    results_dir = pathlib.Path("step_detection_results")
    results_dir.mkdir(exist_ok=True)

    all_metrics = []
    for name, pred in algos.items():
        if pred is None or len(pred) == 0:
            print(f"\nSkipping {name} due to no predictions.")
            continue
        metrics = tester.evaluate_step_detection(pred)
        metrics['algorithm'] = name
        all_metrics.append(metrics)

        print(f"\n{name} metrics:")
        for k, v in metrics.items():
            if k != 'algorithm':
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # produce and save plots + metrics
        tester.plot_processed_signal(peaks=pred)
        tester.save_artifacts(results_dir, name, metrics, pred)

    # Sort the metrics by F1 score and save the final summary report
    if all_metrics:
        sorted_metrics = sorted(all_metrics, key=lambda x: x.get('f1', 0), reverse=True)
        tester.save_summary_report(sorted_metrics, str(results_dir))

# ----------------------------------------------------------------------
# Notes for future improvements:
#  - Use adaptive thresholding with moving percentile + refractory period.
#  - Consider template matched‑filter using average step waveform.
#  - Try bidirectional LSTM on windowed raw signal to predict step onset.
#  - Experiment with TKEO (Teager–Kaiser Energy Operator) + peak‑picking.
#  - For the deep model, explore ensembling techniques (e.g., MC Dropout)
#    and more sophisticated post-processing of predictions.
# ----------------------------------------------------------------------
