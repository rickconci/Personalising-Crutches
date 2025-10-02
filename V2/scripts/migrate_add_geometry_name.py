#!/usr/bin/env python3
"""
Database migration script to add the 'name' column to CrutchGeometry table.

This script:
1. Adds the 'name' column to the crutch_geometries table
2. Populates existing records with generated names based on their parameters
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text, inspect
from database.connection import engine
from database.models import CrutchGeometry
from sqlalchemy.orm import sessionmaker


def check_column_exists(table_name, column_name):
    """Check if a column exists in a table."""
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def main():
    """Run the migration."""
    print("Starting migration: Add 'name' column to CrutchGeometry table...")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        # Check if the column already exists
        if check_column_exists('crutch_geometries', 'name'):
            print("Column 'name' already exists in crutch_geometries table.")
            
            # Check if any records need names populated
            geometries = session.query(CrutchGeometry).all()
            needs_update = False
            
            for geom in geometries:
                if not geom.name or geom.name == '':
                    needs_update = True
                    break
            
            if not needs_update:
                print("All geometry records already have names. Migration not needed.")
                return
            else:
                print("Some records need names populated. Continuing...")
        else:
            # Add the column using raw SQL
            print("Adding 'name' column to crutch_geometries table...")
            with engine.connect() as conn:
                conn.execute(text(
                    "ALTER TABLE crutch_geometries ADD COLUMN name VARCHAR(100)"
                ))
                conn.commit()
            print("Column added successfully.")
        
        # Populate names for existing records
        print("Populating names for existing geometry records...")
        geometries = session.query(CrutchGeometry).all()
        
        if not geometries:
            print("No geometry records found. Migration complete.")
            return
        
        for geom in geometries:
            if not geom.name or geom.name == '':
                # Generate name based on parameters
                if geom.alpha == 95 and geom.beta == 95 and geom.gamma == 0:
                    geom.name = "Control"
                else:
                    geom.name = f"α{geom.alpha}_β{geom.beta}_γ{geom.gamma}"
                print(f"  Updated geometry {geom.id}: {geom.name}")
        
        session.commit()
        print(f"Successfully updated {len(geometries)} geometry records.")
        print("Migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()

