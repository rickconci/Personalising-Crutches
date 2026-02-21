#!/usr/bin/env python3
"""
Add missing columns to trials table for better data tracking
"""
import sqlite3
import sys
from pathlib import Path

# Get the database path
db_path = Path(__file__).parent.parent / "experiments.db"

def add_columns():
    """Add missing columns to trials table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    columns_to_add = [
        ("step_count", "INTEGER"),
        ("instability_loss", "FLOAT"),
        ("laps", "INTEGER"),  # Number of laps completed
    ]
    
    for column_name, column_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE trials ADD COLUMN {column_name} {column_type}")
            print(f"✓ Added column: {column_name} ({column_type})")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"- Column {column_name} already exists, skipping")
            else:
                print(f"✗ Error adding {column_name}: {e}")
                raise
    
    conn.commit()
    conn.close()
    print("\n✓ Migration completed successfully!")

if __name__ == "__main__":
    add_columns()

