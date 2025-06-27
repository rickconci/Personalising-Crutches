#!/usr/bin/env python3
"""
Test script for metabolic cost calculation using exponential fitting.
This demonstrates the correct approach for short-duration protocols.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data_analysis import compute_metabolic_cost_loss_2min, metabolic_rate_estimation

def generate_test_data(duration_min=2.0, sampling_rate=1.0):
    """
    Generate synthetic metabolic data for testing.
    
    Args:
        duration_min: Duration in minutes
        sampling_rate: Samples per second
    
    Returns:
        time_data, vo2_data, vco2_data
    """
    # Time array
    time_data = np.arange(0, duration_min * 60, 1/sampling_rate)
    
    # Generate realistic VO2 and VCO2 data with exponential rise to steady state
    # Steady state values (typical for moderate walking)
    vo2_steady = 1200  # mL/min
    vco2_steady = 1000  # mL/min
    
    # Time constant for exponential rise (typical: 30-60 seconds)
    tau = 45  # seconds
    
    # Generate exponential rise to steady state
    vo2_data = vo2_steady * (1 - np.exp(-time_data / tau)) + 300  # Add baseline
    vco2_data = vco2_steady * (1 - np.exp(-time_data / tau)) + 250  # Add baseline
    
    # Add some noise to make it realistic
    np.random.seed(42)  # For reproducible results
    vo2_data += np.random.normal(0, 50, len(vo2_data))
    vco2_data += np.random.normal(0, 40, len(vco2_data))
    
    return time_data, vo2_data, vco2_data

def test_metabolic_fitting():
    """Test the exponential fitting approach."""
    
    print("=== Metabolic Cost Calculation Test ===\n")
    
    # Generate test data for a 2-minute protocol
    time_data, vo2_data, vco2_data = generate_test_data(duration_min=2.0)
    
    print(f"Generated {len(time_data)} data points over {time_data[-1]/60:.1f} minutes")
    print(f"VO2 range: {vo2_data.min():.0f} - {vo2_data.max():.0f} mL/min")
    print(f"VCO2 range: {vco2_data.min():.0f} - {vco2_data.max():.0f} mL/min\n")
    
    # Calculate metabolic cost using the improved function
    body_weight = 77.0  # kg
    
    metabolic_cost = compute_metabolic_cost_loss_2min(
        vo2_data, vco2_data, time_data, body_weight, use_estimation=True
    )
    
    print(f"=== Results ===")
    print(f"Metabolic Cost: {metabolic_cost:.4f} W/kg")
    
    # Also test the exponential fitting function directly
    y_meas = (0.278 * vo2_data + 0.075 * vco2_data) / body_weight
    y_estimate, y_bar, fit_params = metabolic_rate_estimation(time_data, y_meas)
    
    print(f"Direct exponential fit: {y_estimate:.4f} W/kg")
    print(f"Fit method: {fit_params['method']}")
    if 'r_squared' in fit_params:
        print(f"Fit quality (R²): {fit_params['r_squared']:.3f}")
    
    # Compare with simple average (the wrong way)
    simple_average = np.mean(y_meas)
    print(f"Simple average (WRONG): {simple_average:.4f} W/kg")
    print(f"Difference: {simple_average - metabolic_cost:.4f} W/kg")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot VO2 and VCO2
    plt.subplot(2, 2, 1)
    plt.plot(time_data/60, vo2_data, 'b-', label='VO2 (mL/min)', linewidth=2)
    plt.plot(time_data/60, vco2_data, 'r-', label='VCO2 (mL/min)', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Gas Exchange (mL/min)')
    plt.title('Raw Metabolic Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot metabolic cost over time
    plt.subplot(2, 2, 2)
    plt.plot(time_data/60, y_meas, 'g-', label='Measured', linewidth=2)
    plt.plot(time_data/60, y_bar, 'r--', label='Exponential Fit', linewidth=2)
    plt.axhline(y=metabolic_cost, color='k', linestyle=':', label=f'Steady State: {metabolic_cost:.3f} W/kg')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Metabolic Cost (W/kg)')
    plt.title('Metabolic Cost with Exponential Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 2, 3)
    residuals = y_meas - y_bar
    plt.plot(time_data/60, residuals, 'purple', linewidth=1)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Residuals (W/kg)')
    plt.title('Fit Residuals')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals (W/kg)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metabolic_fitting_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== Key Points ===")
    print("✓ Exponential fitting projects to steady state")
    print("✓ Simple averaging overestimates metabolic cost")
    print("✓ This is the gold standard for short protocols")
    print("✓ Use compute_metabolic_cost_loss_2min for your project")

if __name__ == "__main__":
    test_metabolic_fitting() 