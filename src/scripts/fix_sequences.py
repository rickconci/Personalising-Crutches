#!/usr/bin/env python3
"""
Fix PostgreSQL sequences after data migration.

When data is migrated with explicit IDs, PostgreSQL sequences need to be
updated to prevent duplicate key violations.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import engine, settings
from database.models import (
    Participant, CrutchGeometry, Trial, 
    ExperimentSession, DataFile, OptimizationRun
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# Map of models to their table names and ID columns
TABLES = [
    ("participants", "id"),
    ("crutch_geometries", "id"),
    ("experiment_sessions", "id"),
    ("trials", "id"),
    ("data_files", "id"),
    ("optimization_runs", "id"),
]


def fix_sequence_for_table(session, table_name: str, id_column: str) -> None:
    """Fix the sequence for a specific table."""
    try:
        # Get the maximum ID in the table
        result = session.execute(
            text(f"SELECT COALESCE(MAX({id_column}), 0) FROM {table_name}")
        )
        max_id = result.scalar() or 0
        
        if max_id > 0:
            # Update the sequence to start from max_id + 1
            sequence_name = f"{table_name}_{id_column}_seq"
            session.execute(
                text(f"SELECT setval('{sequence_name}', {max_id}, true)")
            )
            session.commit()
            print(f"✅ Fixed {table_name}: sequence set to {max_id + 1}")
        else:
            print(f"⏭️  {table_name}: No data, sequence unchanged")
    except Exception as e:
        session.rollback()
        print(f"⚠️  {table_name}: {e}")


def main() -> None:
    """Fix all sequences."""
    print("🔧 Fixing PostgreSQL Sequences")
    print("=" * 50)
    
    if "sqlite" in settings.database_url:
        print("⚠️  Using SQLite - sequences not applicable")
        sys.exit(0)
    
    # Create session
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    try:
        print("\n📊 Checking and fixing sequences...\n")
        
        for table_name, id_column in TABLES:
            fix_sequence_for_table(session, table_name, id_column)
        
        print("\n✅ All sequences fixed!")
        print("\nYou can now create new records without ID conflicts.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()

