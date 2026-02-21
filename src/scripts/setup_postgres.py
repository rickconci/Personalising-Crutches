#!/usr/bin/env python3
"""
PostgreSQL setup script for Personalising Crutches.

This script helps set up and initialize PostgreSQL database.
It can be used to:
1. Wait for PostgreSQL to be ready
2. Initialize the database schema
3. Migrate data from SQLite (optional)
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import psycopg2
    from psycopg2 import OperationalError
except ImportError:
    print("❌ psycopg2 is required for PostgreSQL support.")
    print("   Install it with: pip install psycopg2-binary")
    sys.exit(1)

from database.connection import create_tables, engine, settings
from database.models import Participant
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlparse


def check_postgres_connection(max_retries: int = 30, retry_delay: int = 2) -> bool:
    """
    Check if PostgreSQL is accessible.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if connection successful, False otherwise
    """
    if "sqlite" in settings.database_url:
        print("⚠️  Using SQLite, not PostgreSQL. Skipping connection check.")
        return False
    
    parsed = urlparse(settings.database_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")
    user = parsed.username
    password = parsed.password
    
    print(f"🔍 Checking PostgreSQL connection at {host}:{port}...")
    
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                connect_timeout=5
            )
            conn.close()
            print(f"✅ PostgreSQL connection successful!")
            return True
        except OperationalError as e:
            if attempt < max_retries:
                print(f"⏳ Attempt {attempt}/{max_retries} failed. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to connect to PostgreSQL after {max_retries} attempts.")
                print(f"   Error: {e}")
                print("\n💡 Troubleshooting:")
                print("   1. Check your ~/dot_env.txt file or DATABASE_URL in .env")
                print("   2. Verify network connectivity")
                print("   3. Make sure the database server is accessible")
                return False
    
    return False


def initialize_database() -> None:
    """Initialize the database schema."""
    print("\n🗄️  Initializing database schema...")
    try:
        create_tables()
        print("✅ Database tables created successfully.")
        
        # Create session
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        try:
            # Check if participants already exist
            if session.query(Participant).first() is None:
                print("📝 Creating sample participant...")
                sample_participant = Participant(
                    name="Test User",
                    characteristics={
                        "height": 175.0,
                        "weight": 70.0,
                        "forearm_length": 25.0,
                        "fitness_level": 3
                    }
                )
                session.add(sample_participant)
                session.commit()
                print("✅ Sample participant created.")
            else:
                print("ℹ️  Database already contains data, skipping sample creation.")
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise


def main() -> None:
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PostgreSQL setup for Personalising Crutches")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check connection, don't initialize database"
    )
    args = parser.parse_args()
    
    print("🚀 PostgreSQL Setup for Personalising Crutches")
    print("=" * 50)
    
    # Check if using PostgreSQL
    if "sqlite" in settings.database_url:
        if args.check_only:
            # Silent failure for check-only mode
            sys.exit(1)
        print("⚠️  Currently configured for SQLite.")
        print("   To use PostgreSQL, update DATABASE_URL in your .env file.")
        print("   Or set up ~/dot_env.txt with the shared database connection.")
        print("   Example: DATABASE_URL=postgresql://user:pass@localhost:5432/dbname")
        sys.exit(1)
    
    # Check connection
    if not check_postgres_connection():
        sys.exit(1)
    
    # If check-only mode, exit here
    if args.check_only:
        sys.exit(0)
    
    # Initialize database
    try:
        initialize_database()
        print("\n✅ PostgreSQL setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Start the FastAPI server: uvicorn app.main:app --reload")
        print("   2. Access API docs: http://localhost:8000/api/docs")
        print("   3. Access web interface: http://localhost:8000")
        if "localhost" in settings.database_url or "127.0.0.1" in settings.database_url:
            print("   4. Access pgAdmin (if enabled): http://localhost:5050")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

