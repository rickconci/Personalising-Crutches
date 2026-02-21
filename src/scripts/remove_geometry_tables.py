#!/usr/bin/env python3
"""
Migration script to remove predefined geometry tables and columns.

This script removes the crutch_geometries table and the geometry_id column
from the trials table since we're now using dynamic geometry calculation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import engine
from sqlalchemy import text

def remove_geometry_tables():
    """Remove predefined geometry tables and columns."""
    print("Removing predefined geometry tables and columns...")
    
    with engine.connect() as conn:
        # Start transaction
        trans = conn.begin()
        
        try:
            # Remove geometry_id column from trials table
            print("Removing geometry_id column from trials table...")
            conn.execute(text("ALTER TABLE trials DROP COLUMN geometry_id"))
            
            # Drop crutch_geometries table
            print("Dropping crutch_geometries table...")
            conn.execute(text("DROP TABLE IF EXISTS crutch_geometries"))
            
            # Commit transaction
            trans.commit()
            print("✅ Successfully removed geometry tables and columns!")
            
        except Exception as e:
            # Rollback on error
            trans.rollback()
            print(f"❌ Error removing geometry tables: {e}")
            raise

if __name__ == "__main__":
    print("Starting geometry table removal migration...")
    remove_geometry_tables()
    print("Migration completed!")
