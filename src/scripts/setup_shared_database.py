#!/usr/bin/env python3
"""
Setup script to configure shared database connection.

This script helps users set up the shared database by copying the connection
file to their home directory.
"""

import sys
import shutil
from pathlib import Path


def main() -> None:
    """Copy dot_env.txt.example to home directory."""
    print("🔧 Setting up shared database connection...")
    
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    example_file = project_root / "dot_env.txt.example"
    home_dir = Path.home()
    target_file = home_dir / "dot_env.txt"
    
    # Check if example file exists
    if not example_file.exists():
        print(f"❌ Example file not found: {example_file}")
        print("   Make sure you're running this from the V2 directory")
        sys.exit(1)
    
    # Check if target already exists
    if target_file.exists():
        response = input(
            f"⚠️  {target_file} already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if response != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Copy file
    try:
        shutil.copy2(example_file, target_file)
        print(f"✅ Copied database configuration to {target_file}")
        print("\n📋 Next steps:")
        print("   1. The application will automatically use this database")
        print("   2. Run: python scripts/setup_postgres.py")
        print("   3. Start the server: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"❌ Failed to copy file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

