#!/usr/bin/env python3
"""
batch_process_trials.py - Batch process all 3 trials for a participant and create comparison

USAGE:
    python batch_process_trials.py [participant_name]

DESCRIPTION:
    This script processes all 3 trials for a participant and creates comparison plots
    with rankings in one streamlined command.
    
    Example:
        python batch_process_trials.py Friend
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def collect_user_rankings(participant_name, geometries):
    """
    Collect user rankings for the three geometries.
    """
    print(f"\n=== User Ranking Collection for {participant_name} ===")
    print("Please rank the three geometries from MOST stable (1) to LEAST stable (3)")
    print("where 1 = most stable, 2 = medium stability, 3 = least stable")
    
    rankings = {}
    for geometry in geometries:
        while True:
            try:
                rank = int(input(f"Rank for {geometry} (1-3): "))
                if 1 <= rank <= 3:
                    rankings[geometry] = rank
                    break
                else:
                    print("Please enter a number between 1 and 3")
            except ValueError:
                print("Please enter a valid number")
    
    return rankings

def create_comparison_plots(participant_name, geometries, rankings):
    """
    Create comparison plots for the participant.
    """
    print(f"\n=== Creating Comparison Plots for {participant_name} ===")
    
    # Load metrics for all trials
    all_metrics = []
    for i in range(1, 4):
        # Try both naming patterns
        metrics_file = f"{participant_name}{i}_metrics.csv"
        if not os.path.exists(metrics_file):
            metrics_file = f"{participant_name}{i}_data_metrics.csv"
        
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            
            # Extract metrics
            geometry = df[df['metric'] == 'geometry']['value'].iloc[0]
            cycle_duration = float(df[df['metric'] == 'cycle_duration_loss']['value'].iloc[0])
            y_rms = float(df[df['metric'] == 'y_rms_loss']['value'].iloc[0])
            x_rms = float(df[df['metric'] == 'x_rms_loss']['value'].iloc[0])
            
            all_metrics.append({
                'trial': i,
                'geometry': geometry,
                'cycle_duration_loss': cycle_duration,
                'y_rms_loss': y_rms,
                'x_rms_loss': x_rms,
                'user_ranking': rankings[geometry]
            })
    
    if len(all_metrics) != 3:
        print("Error: Need exactly 3 trials for comparison")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create comparison plots - 2x2 layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{participant_name} - Trial Comparison - Instability Metrics', fontsize=20, fontweight='bold')
    fig.patch.set_facecolor('white')

    # Plot 1: User Ratings per Geometry (top left)
    user_ratings = [next(m['user_ranking'] for m in all_metrics if m['geometry'] == g) for g in geometries]
    bars1 = axs[0,0].bar(range(1, 4), user_ratings, color='gold', edgecolor='orange', alpha=0.7)
    for i, bar in enumerate(bars1):
        axs[0,0].annotate(f"{user_ratings[i]}",
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=13)
    axs[0,0].set_title('User Rating per Trial', fontsize=16, fontweight='bold')
    axs[0,0].set_xlabel('Trial Number')
    axs[0,0].set_ylabel('User Rating (1=Most Stable, 3=Least Stable)')
    axs[0,0].set_xticks([1, 2, 3])
    axs[0,0].set_ylim(0.5, 3.5)
    axs[0,0].set_facecolor('white')
    axs[0,0].tick_params(colors='black')
    axs[0,0].yaxis.label.set_color('black')
    axs[0,0].xaxis.label.set_color('black')
    axs[0,0].title.set_color('black')
    axs[0,0].margins(y=0.2)

    # Plot 2: Cycle Duration Variance (top right)
    cycle_vals = [next(m['cycle_duration_loss'] for m in all_metrics if m['geometry'] == g) for g in geometries]
    bars2 = axs[0,1].bar(range(1, 4), cycle_vals, color='skyblue', edgecolor='navy', alpha=0.7)
    for i, bar in enumerate(bars2):
        axs[0,1].annotate(f"{cycle_vals[i]:.4f}",
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=13)
    axs[0,1].set_title('Cycle Duration Variance', fontsize=16, fontweight='bold')
    axs[0,1].set_xlabel('Trial Number')
    axs[0,1].set_ylabel('Standard Deviation (s)')
    axs[0,1].set_xticks([1, 2, 3])
    axs[0,1].set_facecolor('white')
    axs[0,1].tick_params(colors='black')
    axs[0,1].yaxis.label.set_color('black')
    axs[0,1].xaxis.label.set_color('black')
    axs[0,1].title.set_color('black')
    axs[0,1].margins(y=0.2)

    # Plot 3: X-axis RMS per Cycle (bottom left)
    x_rms_vals = [next(m['x_rms_loss'] for m in all_metrics if m['geometry'] == g) for g in geometries]
    bars3 = axs[1,0].bar(range(1, 4), x_rms_vals, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    for i, bar in enumerate(bars3):
        axs[1,0].annotate(f"{x_rms_vals[i]:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=13)
    axs[1,0].set_title('RMS X-axis Acceleration per Cycle', fontsize=16, fontweight='bold')
    axs[1,0].set_xlabel('Trial Number')
    axs[1,0].set_ylabel('RMS (m/s²)')
    axs[1,0].set_xticks([1, 2, 3])
    axs[1,0].set_facecolor('white')
    axs[1,0].tick_params(colors='black')
    axs[1,0].yaxis.label.set_color('black')
    axs[1,0].xaxis.label.set_color('black')
    axs[1,0].title.set_color('black')
    axs[1,0].margins(y=0.2)

    # Plot 4: Y-axis RMS per Cycle (bottom right)
    y_rms_vals = [next(m['y_rms_loss'] for m in all_metrics if m['geometry'] == g) for g in geometries]
    bars4 = axs[1,1].bar(range(1, 4), y_rms_vals, color='salmon', edgecolor='darkred', alpha=0.7)
    for i, bar in enumerate(bars4):
        axs[1,1].annotate(f"{y_rms_vals[i]:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center', va='bottom', fontweight='bold', fontsize=13)
    axs[1,1].set_title('RMS Y-axis Acceleration per Cycle', fontsize=16, fontweight='bold')
    axs[1,1].set_xlabel('Trial Number')
    axs[1,1].set_ylabel('RMS (m/s²)')
    axs[1,1].set_xticks([1, 2, 3])
    axs[1,1].set_facecolor('white')
    axs[1,1].tick_params(colors='black')
    axs[1,1].yaxis.label.set_color('black')
    axs[1,1].xaxis.label.set_color('black')
    axs[1,1].title.set_color('black')
    axs[1,1].margins(y=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = f"{participant_name}_trial_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {output_path}")
    
    # Save rankings
    rankings_file = f"{participant_name}_rankings.csv"
    rankings_df = pd.DataFrame(list(rankings.items()), columns=['geometry', 'ranking'])
    rankings_df['participant'] = participant_name
    rankings_df.to_csv(rankings_file, index=False)
    print(f"User rankings saved to: {rankings_file}")
    
    # Print summary
    print(f"\n=== {participant_name} - Trial Comparison Summary ===")
    print(f"Geometries tested: {', '.join(geometries)}")
    print(f"User rankings: {rankings}")
    
    # Find best geometry by objective metrics
    best_cycle_geometry = metrics_df.loc[metrics_df['cycle_duration_loss'].idxmin(), 'geometry']
    best_cycle_value = metrics_df['cycle_duration_loss'].min()
    print(f"Best stability by cycle duration: {best_cycle_geometry} ({best_cycle_value:.4f} s)")
    
    best_y_rms_geometry = metrics_df.loc[metrics_df['y_rms_loss'].idxmin(), 'geometry']
    best_y_rms_value = metrics_df['y_rms_loss'].min()
    print(f"Best stability by Y-axis RMS: {best_y_rms_geometry} ({best_y_rms_value:.4f} m/s²)")
    
    best_x_rms_geometry = metrics_df.loc[metrics_df['x_rms_loss'].idxmin(), 'geometry']
    best_x_rms_value = metrics_df['x_rms_loss'].min()
    print(f"Best stability by X-axis RMS: {best_x_rms_geometry} ({best_x_rms_value:.4f} m/s²)")

def batch_process_participant(participant_name):
    """
    Process all 3 trials for a participant and create comparison.
    """
    print(f"=== Batch Processing for {participant_name} ===")
    
    # Valid geometry options
    valid_geometries = ["10.5", "14", "18.2"]
    
    # Get geometry order
    print(f"Please provide the geometry order for the 3 trials:")
    print(f"Valid geometries: 10.5, 14, 18.2")
    
    geometry_order = []
    for i in range(3):
        while True:
            geometry = input(f"Geometry for trial {i+1}: ").strip()
            if geometry in valid_geometries:
                geometry_order.append(geometry)
                break
            else:
                print(f"Invalid geometry. Please choose from: {', '.join(valid_geometries)}")
    
    print(f"Geometry order: {geometry_order}")
    
    # Process each trial
    for i in range(1, 4):
        data_file = f"{participant_name}{i}_data.csv"
        geometry = geometry_order[i-1]
        
        if os.path.exists(data_file):
            print(f"\nProcessing trial {i} with geometry: {geometry}")
            
            # Run the data processing script
            cmd = [
                "python", "luke_ble_data_processing.py",
                data_file,
                "--participant", participant_name,
                "--geometry", geometry
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ Successfully processed trial {i}")
                else:
                    print(f"✗ Error processing trial {i}: {result.stderr}")
            except Exception as e:
                print(f"✗ Error running command for trial {i}: {e}")
        else:
            print(f"✗ Data file not found: {data_file}")
    
    # Collect rankings and create comparison
    rankings = collect_user_rankings(participant_name, geometry_order)
    create_comparison_plots(participant_name, geometry_order, rankings)
    
    print(f"\n=== Complete Analysis for {participant_name} ===")
    print("All trials processed, rankings collected, and comparison plots created!")

def main():
    """
    Main function to batch process trials.
    """
    if len(sys.argv) > 1:
        participant_name = sys.argv[1]
    else:
        participant_name = input("Enter participant name: ").strip()
    
    if not participant_name:
        print("Error: Participant name is required.")
        return 1
    
    batch_process_participant(participant_name)
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 