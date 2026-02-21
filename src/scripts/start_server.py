#!/usr/bin/env python3
"""
Simple server start script for Personalising Crutches.

This script just starts the FastAPI server - assumes setup is already done.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the FastAPI server."""
    print("🌐 Starting Personalising Crutches Server...")
    
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
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/api/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the mamba environment: mamba activate opmo2")
        print("2. Run setup first: python scripts/setup_mamba.py")
        print("3. Check if all dependencies are installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
