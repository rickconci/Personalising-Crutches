"""
Migration script to add opencap_events column to trials table.

This script adds the new opencap_events column to support OpenCap toggle functionality.
Run this script once to update your database schema.

Usage:
    python scripts/add_opencap_events_column.py
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path to import database connection
sys.path.insert(0, str(Path(__file__).parent.parent))

def add_opencap_events_column():
    """Add opencap_events column to trials table."""
    
    # Database path
    db_path = Path(__file__).parent.parent / "experiments.db"
    
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("Please make sure experiments.db exists in the V2 directory")
        return False
    
    print(f"📁 Using database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(trials)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'opencap_events' in columns:
            print("✅ Column 'opencap_events' already exists in trials table")
            conn.close()
            return True
        
        # Add the column
        print("🔧 Adding 'opencap_events' column to trials table...")
        cursor.execute("""
            ALTER TABLE trials 
            ADD COLUMN opencap_events JSON DEFAULT NULL
        """)
        
        conn.commit()
        print("✅ Successfully added 'opencap_events' column!")
        
        # Verify the column was added
        cursor.execute("PRAGMA table_info(trials)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'opencap_events' in columns:
            print("✅ Verified: Column exists in database")
        else:
            print("⚠️  Warning: Column may not have been added correctly")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("OpenCap Events Column Migration")
    print("=" * 60)
    print()
    
    success = add_opencap_events_column()
    
    print()
    print("=" * 60)
    
    if success:
        print("✅ Migration completed successfully!")
        print()
        print("Next steps:")
        print("  1. Restart your backend server")
        print("  2. Refresh your frontend")
        print("  3. Start using the OpenCap toggle feature!")
    else:
        print("❌ Migration failed")
        print()
        print("Please check the error messages above and try again")
    
    print("=" * 60)
    
    sys.exit(0 if success else 1)

