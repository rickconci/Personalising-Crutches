import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def metabolic_rate_estimation(time: np.ndarray,
                              y: np.ndarray,
                              tau: float):
    """
    Placeholder for your existing metabolic_rate_estimation routine.
    Should return (y_estimate, y_bar, other).
    """
    # Here you would import or implement your real function.
    # For now, we'll just return a flat line at the mean.
    y_bar = np.convolve(y, np.ones(int(tau))/tau, mode='same')
    return y, y_bar, None

def parse_experiment_time(excel_time) -> str:
    """
    Accept either a pd.Timestamp or a string like '1/1/2025  7:15:30 AM',
    convert to a Timestamp, and return HHMMSS in 24-hour clock.
    """
    ts = pd.to_datetime(excel_time)
    return ts.strftime("%H%M%S")

def _parse_time_value(x):
    """
    Accept one of several possible “time” encodings coming from COSMED / Excel
    and return seconds (float).  Handles:
      • Excel-numeric times (fraction of a day)
      • python datetime / time objects
      • strings like “23:37”  “1:23:45”  “12:05 PM”
    Returns np.nan if it can’t confidently parse.
    """
    import datetime as _dt, numpy as _np, pandas as _pd

    if _pd.isna(x):
        return _np.nan

    # Excel numeric time (0…1 represents 0…24 h)
    if isinstance(x, (int, float)):
        return float(x) * 86400.0

    # Python datetime/time
    if isinstance(x, (_dt.datetime, _dt.time)):
        t = x.time() if isinstance(x, _dt.datetime) else x
        return t.hour*3600 + t.minute*60 + t.second

    # String cases  ➜ strip + normalise
    s = str(x).strip()
    if not s:
        return _np.nan

    # try HH:MM[:SS] possibly with AM/PM
    try:
        # pandas parses a *time‐only* string to Timestamp @ 1970-01-01
        ts = _pd.to_datetime(s, errors='raise').time()
        return ts.hour*3600 + ts.minute*60 + ts.second
    except Exception:
        pass

    # final fallback: manual split
    if ':' in s:
        parts = [float(p) for p in s.replace('PM','').replace('AM','').split(':')]
        if len(parts) == 2:          # MM:SS
            return parts[0]*60 + parts[1]
        if len(parts) == 3:          # HH:MM:SS
            return parts[0]*3600 + parts[1]*60 + parts[2]

    return _np.nan

def _parse_number(x):
    """
    Convert VO2/VCO2 cells that sometimes use a comma decimal (European style)
    or include thousands separators.  Returns float or np.nan.
    """
    import numpy as _np, pandas as _pd
    if _pd.isna(x):
        return _np.nan
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().replace(' ', '')
    # If comma is decimal marker (no dot in string) → replace with dot
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    # If both present and dot is decimal, comma must be thousand-sep → remove
    elif ',' in s and '.' in s and s.find(',') > s.find('.'):
        s = s.replace(',', '')

    try:
        return float(s)
    except ValueError:
        return _np.nan

def data_processing_metabolics(stem: str,
                               subject_weight_kg: float,
                               estimate_threshold_min: float = 4.8,
                               avg_window_min: float = 2,
                               tau: float = 42):
    # 1) read
    df = pd.read_excel(f"{stem}.xlsx", header=None)
    raw_time = df.iloc[1, 4]
    exp_time = parse_experiment_time(raw_time)

    data_raw = df.iloc[:, 9:].copy()
    data_raw.columns = data_raw.iloc[0]
    data = data_raw.iloc[3:].copy()
    # Convert relevant columns to numeric, ignore non-numeric entries
    data['t'] = data['t'].apply(_parse_time_value)
    data['VO2'] = data['VO2'].apply(_parse_number)
    data['VCO2'] = data['VCO2'].apply(_parse_number)

    # ─── DEBUG: inspect parsing results ─────────────────────────────────────────
    print("\n[DEBUG] Parsed `t`, `VO2`, `VCO2` columns (first 10 rows):")
    print(data[['t', 'VO2', 'VCO2']].head(10))
    print("[DEBUG] NaN counts per column:")
    print(data[['t', 'VO2', 'VCO2']].isna().sum())
    print("[DEBUG] dtypes:")
    print(data[['t', 'VO2', 'VCO2']].dtypes)
    # ────────────────────────────────────────────────────────────────────────────

    # Drop any rows where required numeric values are missing
    data = data.dropna(subset=['t', 'VO2', 'VCO2']).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid numeric rows found for t, VO2, and VCO2.")

    # `data['t']` is already in seconds thanks to `_parse_time_value`
    t = data['t'].to_numpy(dtype=float)
    t -= t[0]
    VO2 = data['VO2'].to_numpy(dtype=float)
    VCO2 = data['VCO2'].to_numpy(dtype=float)

    cutoff_idx = np.argmin(np.abs(t - 5*60))
    t = t[:cutoff_idx+1]
    VO2 = VO2[:cutoff_idx+1]
    VCO2 = VCO2[:cutoff_idx+1]

    # use the CLI-provided weight here:
    y_meas = (0.278 * VO2 + 0.075 * VCO2) / subject_weight_kg

    if t[-1] < estimate_threshold_min * 60:
        y_estimate, y_bar, _ = metabolic_rate_estimation(t, y_meas, tau)
        time_bar = t  # y_bar aligns with full t here
        y_average = np.mean(y_meas)
    else:
        start_time = t[-1] - avg_window_min*60
        start_idx = np.argmin(np.abs(t - start_time))
        y_average = np.mean(y_meas[start_idx:])

        end_idx = np.argmin(np.abs(t - 180))
        y_estimate, y_bar, _ = metabolic_rate_estimation(
            t[:end_idx+1], y_meas[:end_idx+1], tau
        )
        t = t[:end_idx+1]
        y_meas = y_meas[:end_idx+1]
        time_bar = t  # y_bar corresponds to this same slice

    print(f"Average: ({exp_time},{int(round(float(t[-1])))}s) {stem} = {y_average:.4f} W/kg")
    if t[-1] >= estimate_threshold_min * 60:
        print(f"Estimate: ({exp_time},{int(round(float(t[-1])))}s) {stem} (est.) = {y_estimate:.4f} W/kg")

    plt.figure(figsize=(8,4))
    plt.plot(t, y_meas, 'o')
    plt.plot(time_bar, y_bar, '-')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Metabolic Cost (W kg$^{-1}$)')
    plt.tight_layout()

    return {
        "exp_time": exp_time,
        "y_average": float(y_average),
        "y_estimate": float(y_estimate) if t[-1] >= estimate_threshold_min*60 else None,
        "figure": plt.gcf()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process metabolics Excel file and save results"
    )
    parser.add_argument(
        "stem",
        nargs="?",                              
        default="/Users/riccardoconci/Library/Mobile Documents/com~apple~CloudDocs/HQ_2024/Projects/2024_Harvard_AIM/Research/OPMO/Personalising-Crutches/2025.06.18/20250618_LukeChung_COSMED",
        help="Input filename without “.xlsx”"
    )
    parser.add_argument(
        "-o", "--outdir",
        default="met_output",
        help="Directory to write plot & JSON"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=77.0,
        help="Subject weight in kg (used to normalize metabolic cost)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing out the PNG figure"
    )
    args = parser.parse_args()

    # pass args.weight into the function:
    results = data_processing_metabolics(
        stem=args.stem,
        subject_weight_kg=args.weight
    )

    if not args.no_plot:
        fig_path = os.path.join(
            args.outdir,
            f"{args.stem}_metabolics.png"
        )
        results["figure"].savefig(fig_path)
        print(f"→ Saved plot: {fig_path}")

    import json
    out_json = {
        "exp_time": results["exp_time"],
        "y_average": results["y_average"],
        "y_estimate": results["y_estimate"]
    }
    json_path = os.path.join(
        args.outdir,
        f"{args.stem}_metabolics.json"
    )
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"→ Saved results: {json_path}")