#!/usr/bin/env python3
"""
Development startup script for Personalising Crutches.

This script sets up the development environment and starts the FastAPI server.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the development server."""
    print("🚀 Starting Personalising Crutches Development Server...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Check if virtual environment exists
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created")
    
    # Determine the correct Python executable
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([str(pip_exe), "install", "-e", "."], check=True)
    print("✅ Dependencies installed")
    
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
        subprocess.run([str(python_exe), "scripts/setup_database.py"], check=True)
        print("✅ Database initialized")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Database initialization failed: {e}")
        print("You may need to run 'python scripts/setup_database.py' manually")
    
    # Start the server
    print("🌐 Starting FastAPI server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/api/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            str(python_exe), "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
