import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit

# Note: Assuming _parse_time_value and _parse_number exist above this line.
# If they don't, they would need to be added.

def metabolic_rate_estimation(time: np.ndarray, y_meas: np.ndarray, tau: float = 42.0):
    """
    Estimate steady-state metabolic cost using exponential rise model.
    This is the gold standard approach for short-duration exercise protocols.
    
    Args:
        time: Time array in seconds
        y_meas: Measured metabolic cost array (W/kg)
        tau: Time constant for exponential fit (default: 42s, typical for moderate exercise)
    
    Returns:
        y_estimate: Estimated steady-state metabolic cost (W/kg)
        y_bar: Fitted exponential curve
        fit_params: Dictionary with fit parameters for debugging
    """
    if len(time) < 10 or len(y_meas) < 10:
        y_bar = np.full_like(y_meas, np.mean(y_meas))
        return np.mean(y_meas), y_bar, {'method': 'simple_average', 'reason': 'insufficient_data'}
    
    def exponential_rise(t, y_ss, tau_fit):
        return y_ss * (1 - np.exp(-t / tau_fit))
    
    try:
        last_30_percent = int(0.3 * len(y_meas))
        y_ss_guess = np.mean(y_meas[-last_30_percent:])
        tau_guess = tau
        
        bounds = ([0.5, 10.0], [50.0, 200.0])
        
        popt, pcov = curve_fit(exponential_rise, time, y_meas, 
                              p0=[y_ss_guess, tau_guess], 
                              bounds=bounds, 
                              maxfev=10000)
        
        y_ss_fitted, tau_fitted = popt
        y_bar = exponential_rise(time, y_ss_fitted, tau_fitted)
        
        fit_params = {
            'method': 'exponential_fit',
            'y_ss': y_ss_fitted,
            'tau': tau_fitted,
            'r_squared': 1 - np.sum((y_meas - y_bar)**2) / np.sum((y_meas - np.mean(y_meas))**2)
        }
        
        return y_ss_fitted, y_bar, fit_params
        
    except (RuntimeError, ValueError) as e:
        print(f"Exponential fitting failed: {e}")
        y_bar = np.full_like(y_meas, np.mean(y_meas))
        return np.mean(y_meas), y_bar, {'method': 'simple_average', 'reason': 'fitting_failed'}

def compute_last_2min_average(vo2_data: np.ndarray, vco2_data: np.ndarray, 
                              time_data: np.ndarray, body_weight_kg: float = 77.0) -> float:
    """
    Calculate metabolic cost using last 2-minute average (gold standard for long protocols).
    """
    if vo2_data.size == 0 or vco2_data.size == 0 or time_data.size == 0 or body_weight_kg <= 0:
        return float('nan')
    
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
    
    last_2min_start = time_data[-1] - 120
    last_2min_mask = time_data >= last_2min_start
    y_last_2min = y_meas[last_2min_mask]
    
    y_average = np.mean(y_last_2min)
    
    print(f"Using gold standard - average of last 2 min: {y_average:.4f} W/kg")
    return y_average

def compute_exponential_estimate(vo2_data: np.ndarray, vco2_data: np.ndarray, 
                                time_data: np.ndarray, body_weight_kg: float = 77.0) -> tuple:
    """
    Calculate metabolic cost using exponential estimation (for short protocols).
    """
    if vo2_data.size == 0 or vco2_data.size == 0 or time_data.size == 0 or body_weight_kg <= 0:
        return float('nan'), {}
    
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
    y_estimate, y_bar, fit_params = metabolic_rate_estimation(time_data, y_meas)
    
    print(f"Using exponential estimation - steady state: {y_estimate:.4f} W/kg")
    print(f"Fit method: {fit_params['method']}")
    if 'r_squared' in fit_params:
        print(f"Fit quality (R²): {fit_params['r_squared']:.3f}")
    
    return y_estimate, fit_params

def compute_metabolic_cost_loss(vo2_data: np.ndarray, vco2_data: np.ndarray, 
                                    time_data: np.ndarray, body_weight_kg: float = 77.0,
                                    use_estimation: bool = True) -> float:
    """
    Compute metabolic cost using Brockway equation with 4.8-minute threshold for protocol classification.
    """
    protocol_duration_min = time_data[-1] / 60.0
    
    if use_estimation and protocol_duration_min < 4.8:
        print(f"Short protocol detected ({protocol_duration_min:.1f} min)")
        y_estimate, _ = compute_exponential_estimate(vo2_data, vco2_data, time_data, body_weight_kg)
        return y_estimate
    
    else:
        if protocol_duration_min >= 4.8:
            print(f"Long protocol detected ({protocol_duration_min:.1f} min)")
            return compute_last_2min_average(vo2_data, vco2_data, time_data, body_weight_kg)
        else:
            y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
            y_average = np.mean(y_meas)
            print(f"Short protocol without estimation ({protocol_duration_min:.1f} min)")
            print(f"Using simple average: {y_average:.4f} W/kg")
            return y_average

def load_raw_metabolic_data(base_path: str, body_weight_kg: float = 77.0) -> tuple:
    """
    Load raw metabolic data (VO2, VCO2, time) from Excel files.
    """
    # This function requires _parse_time_value and _parse_number, assumed to be in this file.
    from .data_processing_metabolic import _parse_time_value, _parse_number

    possible_paths = [
        f"{base_path}_COSMED.xlsx",
        f"{base_path}_COSMED.xls"
    ]
    
    metabolic_excel_path = next((path for path in possible_paths if os.path.exists(path)), None)
    
    if metabolic_excel_path is None:
        print(f"Warning - Raw metabolic data file not found for base path: {base_path}")
        return None, None, None
    
    try:
        df = pd.read_excel(metabolic_excel_path, header=None)
        
        data_raw = df.iloc[:, 9:].copy() 
        data_raw.columns = data_raw.iloc[0]
        data = data_raw.iloc[3:].copy()  
        
        data['t'] = data['t'].apply(_parse_time_value)
        data['VO2'] = data['VO2'].apply(_parse_number)
        data['VCO2'] = data['VCO2'].apply(_parse_number)
        
        data = data.dropna(subset=['t', 'VO2', 'VCO2']).reset_index(drop=True)
        if data.empty:
            return None, None, None
        
        time_data = data['t'].to_numpy(dtype=float)
        vo2_data = data['VO2'].to_numpy(dtype=float)
        vco2_data = data['VCO2'].to_numpy(dtype=float)
        
        time_data -= time_data[0]
        
        cutoff_idx = np.argmin(np.abs(time_data - 5*60))
        time_data = time_data[:cutoff_idx+1]
        vo2_data = vo2_data[:cutoff_idx+1]
        vco2_data = vco2_data[:cutoff_idx+1]
        
        return time_data, vo2_data, vco2_data
        
    except Exception as e:
        print(f"Error loading raw metabolic data: {e}")
        return None, None, None

def process_metabolic_data_complete(base_path: str, body_weight_kg: float = 77.0, 
                                   estimate_threshold_min: float = 4.8, 
                                   avg_window_min: float = 2, tau: float = 42.0):
    """
    Process metabolic data using a 4.8-minute threshold for protocol classification.
    """
    time_data, vo2_data, vco2_data = load_raw_metabolic_data(base_path, body_weight_kg)
    
    if time_data is None:
        return None
    
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight_kg
    protocol_duration_min = time_data[-1] / 60.0
    
    if protocol_duration_min < estimate_threshold_min:
        y_estimate, _ = compute_exponential_estimate(vo2_data, vco2_data, time_data, body_weight_kg)
        y_average, y_bar, _ = metabolic_rate_estimation(time_data, y_meas, tau)
        time_bar = time_data
    else:
        y_average = compute_last_2min_average(vo2_data, vco2_data, time_data, body_weight_kg)
        end_idx = np.argmin(np.abs(time_data - 180))
        time_estimate = time_data[:end_idx+1]
        y_estimate, y_bar, fit_params = metabolic_rate_estimation(
            time_estimate, y_meas[:end_idx+1], tau
        )
        time_bar = time_estimate

    viz_data = {
        'time': time_data, 'y_meas': y_meas, 'time_bar': time_bar, 'y_bar': y_bar,
        'y_average': y_average, 'y_estimate': y_estimate, 'protocol_duration_min': protocol_duration_min,
        'fit_params': fit_params, 'body_weight_kg': body_weight_kg
    }
    
    return viz_data

def get_metabolic_cost_from_excel(base_path: str, body_weight_kg: float = 77.0) -> float:
    """
    Simplified function that always processes metabolic cost from raw Excel data.
    """
    time_data, vo2_data, vco2_data = load_raw_metabolic_data(base_path, body_weight_kg)
    
    if time_data is None:
        print(f"Failed to load raw metabolic data for {base_path}")
        return float('nan')
    
    metabolic_cost = compute_metabolic_cost_loss(vo2_data, vco2_data, time_data, body_weight_kg)
    
    return metabolic_cost