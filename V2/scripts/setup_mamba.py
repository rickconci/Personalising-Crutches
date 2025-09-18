#!/usr/bin/env python3
"""
Setup script for Personalising Crutches with existing mamba environment.

This script assumes you already have mamba activated (mamba activate opmo2).
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Setup the project with existing mamba environment."""
    print("🚀 Setting up Personalising Crutches with mamba environment...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check if we're in a mamba environment
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        print("⚠️  Warning: No conda/mamba environment detected.")
        print("   Please run: mamba activate opmo2")
        print("   Then run this script again.")
        sys.exit(1)
    
    print(f"✅ Using mamba environment: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   You may need to install some packages manually:")
        print("   mamba install fastapi uvicorn sqlalchemy pandas numpy scipy scikit-learn matplotlib seaborn plotly")
        sys.exit(1)
    
    # Create necessary directories
    print("📁 Creating directories...")
    directories = [
        "data",
        "data/raw",
        "data/processed", 
        "data/results",
        "data/plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created")
    
    # Initialize database
    print("🗄️ Initializing database...")
    try:
        subprocess.run([sys.executable, "scripts/setup_database.py"], check=True)
        print("✅ Database initialized")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Database initialization failed: {e}")
        print("You may need to run 'python scripts/setup_database.py' manually")
    
    print("\n🎉 Setup complete!")
    print("\nTo start the server, run:")
    print("  uvicorn app.main:app --reload")
    print("\nThen visit:")
    print("  🌐 Web Interface: http://localhost:8000")
    print("  📚 API Docs: http://localhost:8000/api/docs")

if __name__ == "__main__":
    main()
