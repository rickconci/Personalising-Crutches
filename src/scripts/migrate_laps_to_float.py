"""
Migration script to change laps column from Integer to Float
"""

import sqlite3
import os

# Path to the database
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments.db')

def migrate_laps_to_float():
    """Change laps column type from INTEGER to REAL (Float)"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        print("Starting migration: laps column INTEGER -> REAL")
        
        # SQLite doesn't support ALTER COLUMN TYPE directly
        # We need to recreate the table
        
        # 1. Create a temporary table with the new schema
        cursor.execute("""
            CREATE TABLE trials_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_id INTEGER NOT NULL,
                geometry_id INTEGER,
                session_id INTEGER,
                alpha REAL,
                beta REAL,
                gamma REAL,
                delta REAL,
                source TEXT,
                raw_data_path TEXT,
                processed_features JSON,
                survey_responses JSON,
                steps JSON,
                
                sus_q1 INTEGER,
                sus_q2 INTEGER,
                sus_q3 INTEGER,
                sus_q4 INTEGER,
                sus_q5 INTEGER,
                sus_q6 INTEGER,
                sus_score REAL,
                nrs_score INTEGER,
                tlx_mental_demand INTEGER,
                tlx_physical_demand INTEGER,
                tlx_performance INTEGER,
                tlx_effort INTEGER,
                tlx_frustration INTEGER,
                tlx_score INTEGER,
                
                metabolic_cost REAL,
                y_change REAL,
                y_total REAL,
                step_count INTEGER,
                step_variance REAL,
                instability_loss REAL,
                rms_load_cell_force REAL,
                total_combined_loss REAL,
                laps REAL,
                
                deleted_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (participant_id) REFERENCES participants(id),
                FOREIGN KEY (geometry_id) REFERENCES crutch_geometries(id),
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(id)
            )
        """)
        
        # 2. Copy data from old table to new table
        cursor.execute("""
            INSERT INTO trials_new
            SELECT * FROM trials
        """)
        
        # 3. Drop old table
        cursor.execute("DROP TABLE trials")
        
        # 4. Rename new table to original name
        cursor.execute("ALTER TABLE trials_new RENAME TO trials")
        
        # 5. Recreate indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_participant ON trials(participant_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_geometry ON trials(geometry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_deleted ON trials(deleted_at)")
        
        conn.commit()
        print("✓ Migration completed successfully")
        print("  - laps column is now REAL (Float) type")
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_laps_to_float()

