#!/usr/bin/env python3
"""
Quick script to extract steps data from experiments.db
Usage: python quick_steps_query.py
"""

import pandas as pd
import sqlite3
from pathlib import Path

# Database path
db_path = Path(__file__).parent.parent / "experiments.db"

# Simple query to get steps data
query = """
SELECT 
    id as trial_id,
    participant_id,
    steps,
    step_count,
    step_variance,
    timestamp,
    alpha,
    beta,
    gamma,
    delta
FROM trials 
WHERE steps IS NOT NULL
ORDER BY trial_id
"""

# Execute query
conn = sqlite3.connect(str(db_path))
df = pd.read_sql_query(query, conn)
conn.close()

# Display results
print(f"Found {len(df)} trials with steps data")
print(f"Participants: {df['participant_id'].nunique()}")
print(f"Average steps per trial: {df['step_count'].mean():.1f}")
print(f"Average step variance: {df['step_variance'].mean():.4f}")

# Show first few rows
print("\nFirst 5 trials:")
print(df[['trial_id', 'participant_id', 'step_count', 'step_variance']].head())

# Save to CSV
output_file = Path(__file__).parent.parent / "steps_data_export.csv"
df.to_csv(output_file, index=False)
print(f"\nData saved to: {output_file}")

# Show steps data for first trial
if len(df) > 0:
    print(f"\nSteps data for trial {df.iloc[0]['trial_id']}:")
    steps_data = df.iloc[0]['steps']
    if steps_data:
        import json
        try:
            steps_list = json.loads(steps_data) if isinstance(steps_data, str) else steps_data
            print(f"Number of steps: {len(steps_list)}")
            print(f"First 5 step times: {steps_list[:5]}")
        except:
            print("Could not parse steps data")

