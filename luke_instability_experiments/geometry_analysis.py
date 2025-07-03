#!/usr/bin/env python3
"""
Analyze geometry rankings across all participants.
Calculate average instability rating for each geometry and trial order correlation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy import stats
import re

def load_all_rankings():
    """
    Load all participant ranking files and combine them.
    
    Returns:
        DataFrame with columns: participant, geometry, ranking
    """
    # Find all ranking files
    ranking_files = glob.glob("*_rankings.csv")
    
    all_rankings = []
    for file in ranking_files:
        df = pd.read_csv(file)
        all_rankings.append(df)
    
    if not all_rankings:
        print("No ranking files found!")
        return None
    
    # Combine all rankings
    combined_df = pd.concat(all_rankings, ignore_index=True)
    return combined_df

def load_trial_order_data():
    """
    Load trial order data from metrics files and match to user rankings.
    Returns DataFrame with participant, trial_number, geometry, ranking columns.
    """
    metrics_files = glob.glob("*_data_metrics.csv")
    print(f"Found {len(metrics_files)} metrics files")
    
    # Load all rankings into a lookup dict: (participant, geometry) -> ranking
    all_rankings = load_all_rankings()
    ranking_lookup = {(row['participant'], float(row['geometry'])): row['ranking'] for _, row in all_rankings.iterrows()}
    
    trial_data = []
    for file in metrics_files:
        filename = os.path.basename(file)
        match = re.match(r'(\D+)(\d+)_data_metrics\.csv', filename)
        if match:
            participant = match.group(1)
            trial_num = int(match.group(2))
            try:
                df = pd.read_csv(file)
                geometry = float(df[df['metric'] == 'geometry']['value'].iloc[0])
                ranking = ranking_lookup.get((participant, geometry), None)
                trial_data.append({
                    'participant': participant,
                    'trial_number': trial_num,
                    'geometry': geometry,
                    'ranking': ranking
                })
                print(f"  Loaded: {filename} -> {participant} trial {trial_num} geometry {geometry} ranking {ranking}")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
        else:
            print(f"  Skipped: {filename} (no match)")
    print(f"Loaded {len(trial_data)} trial records")
    return pd.DataFrame(trial_data)

def calculate_average_rankings(rankings_df):
    """
    Calculate average ranking for each geometry.
    
    Args:
        rankings_df: DataFrame with participant, geometry, ranking columns
        
    Returns:
        DataFrame with geometry and average_ranking columns
    """
    # Group by geometry and calculate mean ranking
    avg_rankings = rankings_df.groupby('geometry')['ranking'].agg(['mean', 'std', 'count']).reset_index()
    avg_rankings.columns = ['geometry', 'average_ranking', 'std_ranking', 'participant_count']
    
    # Sort by geometry (numerical order)
    avg_rankings['geometry_numeric'] = avg_rankings['geometry'].astype(float)
    avg_rankings = avg_rankings.sort_values('geometry_numeric').drop('geometry_numeric', axis=1)
    
    return avg_rankings

def create_geometry_comparison_plot(avg_rankings_df):
    """
    Create bar plot comparing average rankings across geometries.
    
    Args:
        avg_rankings_df: DataFrame with geometry and average_ranking columns
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bar plot
    geometries = avg_rankings_df['geometry'].astype(str)
    avg_rankings = avg_rankings_df['average_ranking']
    std_rankings = avg_rankings_df['std_ranking']
    participant_counts = avg_rankings_df['participant_count']
    
    bars = ax.bar(geometries, avg_rankings, 
                  capsize=5, 
                  color='skyblue', 
                  edgecolor='navy', 
                  alpha=0.7,
                  width=0.6)
    
    # Add value labels on bars with ± notation
    for i, (bar, avg, std, count) in enumerate(zip(bars, avg_rankings, std_rankings, participant_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{avg:.2f}±{std:.2f}\n(n={int(count)})',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Customize plot
    ax.set_title('Average User Instability Rating by Geometry', fontsize=18, fontweight='bold')
    ax.set_xlabel('Geometry (cm)', fontsize=14)
    ax.set_ylabel('Average Rating (1=Most Stable, 3=Least Stable)', fontsize=14)
    ax.set_ylim(0.5, 3.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('white')
    
    # Add horizontal line at y=2 (neutral point)
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Neutral (2.0)')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_trial_order_correlation_plot(trial_order_df):
    """
    Create bar plot showing average rating by trial number.
    
    Args:
        trial_order_df: DataFrame with trial_number and ranking columns
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average rating for each trial number
    trial_avg = trial_order_df.groupby('trial_number')['ranking'].agg(['mean', 'std', 'count']).reset_index()
    trial_avg.columns = ['trial_number', 'average_rating', 'std_rating', 'trial_count']
    
    # Create bar plot
    trial_numbers = trial_avg['trial_number'].astype(str)
    avg_ratings = trial_avg['average_rating']
    std_ratings = trial_avg['std_rating']
    trial_counts = trial_avg['trial_count']
    
    bars = ax.bar(trial_numbers, avg_ratings, 
                  color='lightcoral', 
                  edgecolor='darkred', 
                  alpha=0.7,
                  width=0.6)
    
    # Add value labels on bars with ± notation
    for i, (bar, avg, std, count) in enumerate(zip(bars, avg_ratings, std_ratings, trial_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{avg:.2f}±{std:.2f}\n(n={int(count)})',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Customize plot
    ax.set_title('Average User Rating by Trial Number', fontsize=18, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=14)
    ax.set_ylabel('Average Rating (1=Most Stable, 3=Least Stable)', fontsize=14)
    ax.set_ylim(0.5, 3.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('white')
    
    # Add horizontal line at y=2 (neutral point)
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Neutral (2.0)')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_combined_plots(avg_rankings_df, trial_order_df):
    """
    Create side-by-side plots for geometry analysis.
    
    Args:
        avg_rankings_df: DataFrame with average rankings
        trial_order_df: DataFrame with trial order data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Geometry Analysis: Average Rankings and Trial Order Correlation', fontsize=20, fontweight='bold')
    
    # Plot 1: Average rankings by geometry
    geometries = avg_rankings_df['geometry'].astype(str)
    avg_rankings = avg_rankings_df['average_ranking']
    std_rankings = avg_rankings_df['std_ranking']
    participant_counts = avg_rankings_df['participant_count']
    
    bars = ax1.bar(geometries, avg_rankings, 
                   capsize=5, 
                   color='skyblue', 
                   edgecolor='navy', 
                   alpha=0.7,
                   width=0.6)
    
    # Add value labels on bars with ± notation
    for i, (bar, avg, std, count) in enumerate(zip(bars, avg_rankings, std_rankings, participant_counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{avg:.2f}±{std:.2f}\n(n={int(count)})',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_title('Average User Instability Rating by Geometry', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Geometry (cm)', fontsize=14)
    ax1.set_ylabel('Average Rating (1=Most Stable, 3=Least Stable)', fontsize=14)
    ax1.set_ylim(0.5, 3.5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('white')
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Neutral (2.0)')
    ax1.legend()
    
    # Plot 2: Trial order correlation
    trial_avg = trial_order_df.groupby('trial_number')['ranking'].agg(['mean', 'std', 'count']).reset_index()
    trial_avg.columns = ['trial_number', 'average_rating', 'std_rating', 'trial_count']
    
    trial_numbers = trial_avg['trial_number'].astype(str)
    avg_ratings = trial_avg['average_rating']
    std_ratings = trial_avg['std_rating']
    trial_counts = trial_avg['trial_count']
    
    bars2 = ax2.bar(trial_numbers, avg_ratings, 
                    color='lightcoral', 
                    edgecolor='darkred', 
                    alpha=0.7,
                    width=0.6)
    
    # Add value labels on bars with ± notation
    for i, (bar, avg, std, count) in enumerate(zip(bars2, avg_ratings, std_ratings, trial_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{avg:.2f}±{std:.2f}\n(n={int(count)})',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_title('Average User Rating by Trial Number', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Trial Number', fontsize=14)
    ax2.set_ylabel('Average Rating (1=Most Stable, 3=Least Stable)', fontsize=14)
    ax2.set_ylim(0.5, 3.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('white')
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Neutral (2.0)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('geometry_analysis.png', dpi=300, bbox_inches='tight')
    print("Combined geometry analysis plot saved to: geometry_analysis.png")
    
    return fig

def print_summary_statistics(avg_rankings_df, rankings_df, trial_order_df):
    """
    Print summary statistics about the rankings and trial order correlation.
    
    Args:
        avg_rankings_df: DataFrame with average rankings
        rankings_df: Original rankings DataFrame
        trial_order_df: Trial order DataFrame
    """
    print("\n=== Geometry Analysis Summary ===")
    print(f"Total participants: {rankings_df['participant'].nunique()}")
    print(f"Total ratings: {len(rankings_df)}")
    
    print("\nAverage Rankings by Geometry:")
    for _, row in avg_rankings_df.iterrows():
        geometry = row['geometry']
        avg_rating = row['average_ranking']
        std_rating = row['std_ranking']
        count = int(row['participant_count'])
        print(f"  {geometry} cm: {avg_rating:.2f} ± {std_rating:.2f} (n={count})")
    
    # Find best and worst geometries
    best_geometry = avg_rankings_df.loc[avg_rankings_df['average_ranking'].idxmin()]
    worst_geometry = avg_rankings_df.loc[avg_rankings_df['average_ranking'].idxmax()]
    
    print(f"\nMost Stable Geometry: {best_geometry['geometry']} cm (rating: {best_geometry['average_ranking']:.2f})")
    print(f"Least Stable Geometry: {worst_geometry['geometry']} cm (rating: {worst_geometry['average_ranking']:.2f})")
    
    # Trial order correlation analysis
    if len(trial_order_df) > 0:
        trial_avg = trial_order_df.groupby('trial_number')['ranking'].agg(['mean', 'std', 'count']).reset_index()
        trial_avg.columns = ['trial_number', 'average_rating', 'std_rating', 'trial_count']
        
        print(f"\nTrial Order Analysis:")
        for _, row in trial_avg.iterrows():
            trial_num = int(row['trial_number'])
            avg_rating = row['average_rating']
            std_rating = row['std_rating']
            count = int(row['trial_count'])
            print(f"  Trial {trial_num}: {avg_rating:.2f} ± {std_rating:.2f} (n={count})")
        
        # Check for trends
        if len(trial_avg) > 1:
            first_trial = trial_avg.iloc[0]['average_rating']
            last_trial = trial_avg.iloc[-1]['average_rating']
            if last_trial > first_trial:
                print(f"  Trend: Later trials rated as more unstable (+{last_trial - first_trial:.2f})")
            elif last_trial < first_trial:
                print(f"  Trend: Later trials rated as more stable (-{first_trial - last_trial:.2f})")
            else:
                print(f"  Trend: No clear trend in trial order")
    else:
        print(f"\nTrial Order Analysis: No trial order data available")
    
    # Show individual participant preferences
    print("\nIndividual Participant Rankings:")
    for participant in sorted(rankings_df['participant'].unique()):
        participant_data = rankings_df[rankings_df['participant'] == participant]
        rankings_str = ", ".join([f"{row['geometry']}cm={row['ranking']}" for _, row in participant_data.iterrows()])
        print(f"  {participant}: {rankings_str}")

def main():
    """
    Main function to analyze geometry rankings and trial order correlation.
    """
    print("=== Geometry Analysis ===")
    
    # Load all rankings
    rankings_df = load_all_rankings()
    if rankings_df is None:
        return 1
    
    print(f"Loaded rankings from {len(rankings_df)} participants")
    
    # Load trial order data
    trial_order_df = load_trial_order_data()
    if len(trial_order_df) > 0:
        # Filter out trials with None rankings
        trial_order_df = trial_order_df.dropna(subset=['ranking'])
        print(f"Loaded trial order data for {len(trial_order_df)} trials (after filtering None rankings)")
    else:
        print("Warning: No trial order data found")
    
    # Calculate average rankings
    avg_rankings_df = calculate_average_rankings(rankings_df)
    
    # Create combined visualization
    if len(trial_order_df) > 0:
        create_combined_plots(avg_rankings_df, trial_order_df)
    else:
        # Fallback to just the average rankings plot
        fig = create_geometry_comparison_plot(avg_rankings_df)
        plt.savefig('geometry_analysis.png', dpi=300, bbox_inches='tight')
        print("Geometry analysis plot saved to: geometry_analysis.png")
    
    # Print summary statistics
    print_summary_statistics(avg_rankings_df, rankings_df, trial_order_df)
    
    # Save summary data
    avg_rankings_df.to_csv('geometry_analysis.csv', index=False)
    print("\nSummary data saved to: geometry_analysis.csv")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 