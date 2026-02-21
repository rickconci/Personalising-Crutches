#!/usr/bin/env python3
"""
Migration script to move data from SQLite to PostgreSQL.

This script helps migrate existing SQLite database data to PostgreSQL.
It preserves all participants, trials, geometries, and related data.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, make_transient
from sqlalchemy.pool import StaticPool
from sqlalchemy.inspection import inspect

from database.models import (
    Base, Participant, CrutchGeometry, Trial, 
    ExperimentSession, DataFile, OptimizationRun
)

# Models to migrate (in dependency order)
MODELS = [
    Participant,
    CrutchGeometry,
    ExperimentSession,
    Trial,
    DataFile,
    OptimizationRun,
]


def get_sqlite_session(sqlite_path: str) -> Session:
    """Create a session for SQLite database."""
    sqlite_url = f"sqlite:///{sqlite_path}"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def get_postgres_session(postgres_url: str) -> Session:
    """Create a session for PostgreSQL database."""
    engine = create_engine(postgres_url)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def migrate_table(
    source_session: Session,
    target_session: Session,
    model_class: type,
    model_name: str
) -> int:
    """
    Migrate a single table from source to target database.
    
    Args:
        source_session: Source database session
        target_session: Target database session
        model_class: SQLAlchemy model class
        model_name: Human-readable name for logging
        
    Returns:
        Number of records migrated
    """
    print(f"  Migrating {model_name}...", end=" ", flush=True)
    
    try:
        # Get all records from source
        records = source_session.query(model_class).all()
        
        if not records:
            print("(empty)")
            return 0
        
        # Check if records already exist in target (to avoid duplicates)
        existing_ids = set()
        if hasattr(model_class, 'id'):
            existing = target_session.query(model_class.id).all()
            existing_ids = {row[0] for row in existing}
        
        migrated_count = 0
        
        # Migrate each record
        for record in records:
            # Skip if already exists (based on ID)
            if hasattr(record, 'id') and record.id in existing_ids:
                continue
            
            # Detach the object from source session
            source_session.expunge(record)
            make_transient(record)
            
            # Preserve the ID to maintain foreign key relationships
            # PostgreSQL will use the ID if the sequence allows it
            # If there's a conflict, we'll handle it in the exception
            
            # Add to target session
            target_session.add(record)
            migrated_count += 1
        
        if migrated_count > 0:
            try:
                target_session.commit()
                print(f"✅ {migrated_count} records (skipped {len(records) - migrated_count} duplicates)")
            except Exception as commit_error:
                # If commit fails (e.g., ID conflicts), try with auto-assigned IDs
                target_session.rollback()
                print(f"\n   ⚠️  Commit failed: {commit_error}")
                print(f"   Retrying with auto-assigned IDs...")
                
                # Re-query records since we expunged them
                records_to_migrate = source_session.query(model_class).filter(
                    ~model_class.id.in_(existing_ids) if hasattr(model_class, 'id') else True
                ).all()
                
                # Retry migration with auto-assigned IDs
                migrated_count = 0
                for record in records_to_migrate:
                    source_session.expunge(record)
                    make_transient(record)
                    
                    # Reset ID to let PostgreSQL auto-assign
                    if hasattr(record, 'id'):
                        record.id = None
                    
                    target_session.add(record)
                    migrated_count += 1
                
                if migrated_count > 0:
                    target_session.commit()
                    print(f"✅ {migrated_count} records migrated (with new IDs)")
                else:
                    print(f"⏭️  No new records to migrate")
        else:
            print(f"⏭️  All {len(records)} records already exist (skipped)")
        
        return migrated_count
        
    except Exception as e:
        target_session.rollback()
        print(f"❌ Error: {e}")
        raise


def get_postgres_url_from_home_file() -> Optional[str]:
    """Get PostgreSQL URL from ~/dot_env.txt file."""
    home_dir = Path.home()
    dot_env_file = home_dir / "dot_env.txt"
    
    if not dot_env_file.exists():
        return None
    
    try:
        with open(dot_env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("external db url:"):
                    url = line.split(":", 1)[1].strip()
                    if url:
                        return url
    except Exception:
        pass
    
    return None


def main() -> None:
    """Main migration function."""
    print("🔄 SQLite to PostgreSQL Migration")
    print("=" * 50)
    
    # Get SQLite path
    project_root = Path(__file__).parent.parent
    default_sqlite = project_root / "experiments.db"
    
    sqlite_path = input(f"Enter path to SQLite database [{default_sqlite}]: ").strip()
    if not sqlite_path:
        sqlite_path = str(default_sqlite)
    
    sqlite_path = Path(sqlite_path).resolve()
    
    if not sqlite_path.exists():
        print(f"❌ SQLite database not found: {sqlite_path}")
        sys.exit(1)
    
    # Try to get PostgreSQL URL from ~/dot_env.txt first
    postgres_url = get_postgres_url_from_home_file()
    
    if postgres_url:
        print(f"✅ Found PostgreSQL URL in ~/dot_env.txt")
        use_auto = input(f"Use this database? [Y/n]: ").strip().lower()
        if use_auto and use_auto != 'y':
            postgres_url = None
    
    if not postgres_url:
        postgres_url = input(
            "Enter PostgreSQL connection URL: "
        ).strip()
        
        if not postgres_url:
            print("❌ PostgreSQL URL is required")
            sys.exit(1)
    
    # Confirm
    print(f"\n📋 Migration plan:")
    print(f"   Source: {sqlite_path}")
    print(f"   Target: {postgres_url}")
    confirm = input("\n⚠️  This will copy all data to PostgreSQL. Continue? [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("Migration cancelled.")
        sys.exit(0)
    
    # Create sessions
    print("\n🔌 Connecting to databases...")
    try:
        source_session = get_sqlite_session(str(sqlite_path))
        target_session = get_postgres_session(postgres_url)
        print("✅ Connected to both databases")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)
    
    # Create tables in PostgreSQL
    print("\n🗄️  Creating tables in PostgreSQL...")
    try:
        Base.metadata.create_all(bind=target_session.bind)
        print("✅ Tables created")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        sys.exit(1)
    
    # Migrate data
    print("\n📦 Migrating data...")
    total_records = 0
    
    try:
        for model_class in MODELS:
            model_name = model_class.__name__
            count = migrate_table(source_session, target_session, model_class, model_name)
            total_records += count
        
        print(f"\n✅ Migration completed! Total records migrated: {total_records}")
        print("\n📋 Next steps:")
        print("   1. Update your .env file with the PostgreSQL DATABASE_URL")
        print("   2. Test the connection: python scripts/setup_postgres.py")
        print("   3. Start the server and verify data is accessible")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        target_session.rollback()
        sys.exit(1)
    finally:
        source_session.close()
        target_session.close()


if __name__ == "__main__":
    main()

