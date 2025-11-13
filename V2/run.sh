#!/bin/bash
# Run script for Personalising Crutches
# This script:
# 1. Checks/creates Python environment with dependencies
# 2. Checks database connection
# 3. Initializes database if needed
# 4. Starts the uvicorn server

# Don't use set -e, we want to handle errors gracefully

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}🚀 Personalising Crutches - Starting Server${NC}"
echo "=========================================="

# Change to project directory
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo -e "${GREEN}✅ Using Python: $($PYTHON_CMD --version)${NC}"

# Check for Python environment and dependencies
echo ""
echo -e "${BLUE}🔍 Checking Python environment and dependencies...${NC}"

USE_VENV=false
PYTHON_ENV_CMD=""
CONDA_ENV_FOUND=false

# Check if current Python is in a conda/mamba environment
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX" ]; then
    # Already in a conda environment
    CONDA_ENV_PATH="$CONDA_PREFIX"
    CONDA_ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename $CONDA_PREFIX)}"
    CONDA_ENV_FOUND=true
    echo -e "${GREEN}✅ Detected active conda/mamba environment: ${CONDA_ENV_NAME}${NC}"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    # Environment name is set, try to find the path
    if command -v conda &> /dev/null || command -v mamba &> /dev/null; then
        MAMBA_CMD=""
        if command -v mamba &> /dev/null; then
            MAMBA_CMD="mamba"
        elif command -v conda &> /dev/null; then
            MAMBA_CMD="conda"
        fi
        
        if [ -n "$MAMBA_CMD" ]; then
            # Source conda/mamba initialization
            CONDA_INIT=""
            if [ -f ~/miniforge3/etc/profile.d/conda.sh ]; then
                CONDA_INIT=~/miniforge3/etc/profile.d/conda.sh
            elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
                CONDA_INIT=~/anaconda3/etc/profile.d/conda.sh
            elif [ -f /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh ]; then
                CONDA_INIT=/opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
            fi
            
            if [ -n "$CONDA_INIT" ]; then
                source "$CONDA_INIT" 2>/dev/null
            fi
            
            ENV_PATH=$($MAMBA_CMD env list 2>/dev/null | grep -E "^\s*${CONDA_DEFAULT_ENV}\s" | awk '{print $NF}' | head -1)
            if [ -n "$ENV_PATH" ] && [ -d "$ENV_PATH" ]; then
                CONDA_ENV_PATH="$ENV_PATH"
                CONDA_ENV_NAME="$CONDA_DEFAULT_ENV"
                CONDA_ENV_FOUND=true
                echo -e "${GREEN}✅ Found conda/mamba environment: ${CONDA_ENV_NAME}${NC}"
            fi
        fi
    fi
fi

# Check if Python executable is in a conda environment path
if [ "$CONDA_ENV_FOUND" = false ]; then
    PYTHON_PATH=$(which $PYTHON_CMD 2>/dev/null || command -v $PYTHON_CMD 2>/dev/null)
    if [ -n "$PYTHON_PATH" ]; then
        # Check if Python is in a conda environment (common paths)
        PYTHON_DIR=$(dirname "$PYTHON_PATH")
        ENV_ROOT=$(dirname "$PYTHON_DIR")
        if echo "$PYTHON_PATH" | grep -qE "(conda|mamba|miniforge|anaconda)" && [ -d "$ENV_ROOT/conda-meta" ] 2>/dev/null; then
            CONDA_ENV_PATH="$ENV_ROOT"
            CONDA_ENV_NAME=$(basename "$CONDA_ENV_PATH")
            CONDA_ENV_FOUND=true
            echo -e "${GREEN}✅ Detected conda/mamba environment from Python path: ${CONDA_ENV_NAME}${NC}"
        fi
    fi
fi

# If we found a conda environment, check and install dependencies
if [ "$CONDA_ENV_FOUND" = true ]; then
    # Get Python executable from the conda environment
    CONDA_PYTHON="$CONDA_ENV_PATH/bin/$PYTHON_CMD"
    if [ ! -f "$CONDA_PYTHON" ]; then
        CONDA_PYTHON="$CONDA_ENV_PATH/bin/python"
    fi
    
    if [ -f "$CONDA_PYTHON" ]; then
        # Check if critical dependencies are installed
        echo -e "${BLUE}   Checking dependencies...${NC}"
        DEPS_OK=true
        
        # Test critical packages
        if ! "$CONDA_PYTHON" -c "import fastapi" 2>/dev/null; then
            DEPS_OK=false
        elif ! "$CONDA_PYTHON" -c "import uvicorn" 2>/dev/null; then
            DEPS_OK=false
        elif ! "$CONDA_PYTHON" -c "import sqlalchemy" 2>/dev/null; then
            DEPS_OK=false
        elif ! "$CONDA_PYTHON" -c "import pydantic" 2>/dev/null; then
            DEPS_OK=false
        fi
        
        if [ "$DEPS_OK" = false ]; then
            echo -e "${YELLOW}⚠️  Dependencies are missing from your conda/mamba environment${NC}"
            echo ""
            read -p "$(echo -e ${YELLOW}Would you like to install all dependencies from pyproject.toml? [y/N]: ${NC})" -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}📥 Installing dependencies from pyproject.toml...${NC}"
                "$CONDA_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet
                # Install dependencies directly without editable mode to avoid package discovery issues
                "$CONDA_PYTHON" << 'INSTALL_EOF'
import sys
import re
from pathlib import Path
import subprocess

def extract_dependencies():
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return []
    
    try:
        try:
            import tomllib
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
        except ImportError:
            try:
                import tomli as tomllib
                with open(pyproject_path, 'rb') as f:
                    data = tomllib.load(f)
            except ImportError:
                raise ImportError("No TOML parser available")
        
        return data.get('project', {}).get('dependencies', [])
    except (ImportError, KeyError):
        # Fallback: manual parsing
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        dependencies = []
        in_dependencies = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('dependencies = ['):
                in_dependencies = True
                continue
            elif in_dependencies:
                if line.startswith(']'):
                    break
                match = re.search(r'["\']([^"\']+)["\']', line)
                if match:
                    dependencies.append(match.group(1))
        return dependencies

deps = extract_dependencies()
if deps:
    print(f"Installing {len(deps)} packages...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + deps)
INSTALL_EOF
                echo -e "${GREEN}✅ Dependencies installed${NC}"
            else
                echo -e "${YELLOW}⚠️  Skipping installation. The server may not work correctly without these packages.${NC}"
            fi
        else
            echo -e "${GREEN}✅ Dependencies found in conda/mamba environment${NC}"
        fi
        
        PYTHON_ENV_CMD="$CONDA_PYTHON"
    else
        echo -e "${YELLOW}⚠️  Could not find Python in conda environment${NC}"
        USE_VENV=true
    fi
else
    echo -e "${YELLOW}⚠️  No conda/mamba environment detected${NC}"
    USE_VENV=true
fi

# Create and use .venv if needed
if [ "$USE_VENV" = true ]; then
    if [ ! -d ".venv" ]; then
        echo -e "${BLUE}📦 Creating virtual environment...${NC}"
        $PYTHON_CMD -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Use the venv's Python explicitly
    VENV_PYTHON=".venv/bin/$PYTHON_CMD"
    if [ ! -f "$VENV_PYTHON" ]; then
        VENV_PYTHON=".venv/bin/python"
    fi
    
    # Check if critical dependencies are installed
    echo -e "${BLUE}   Checking dependencies...${NC}"
    DEPS_OK=true
    
    # Test critical packages
    if ! "$VENV_PYTHON" -c "import fastapi" 2>/dev/null; then
        DEPS_OK=false
    elif ! "$VENV_PYTHON" -c "import uvicorn" 2>/dev/null; then
        DEPS_OK=false
    elif ! "$VENV_PYTHON" -c "import sqlalchemy" 2>/dev/null; then
        DEPS_OK=false
    elif ! "$VENV_PYTHON" -c "import pydantic" 2>/dev/null; then
        DEPS_OK=false
    fi
    
    if [ "$DEPS_OK" = false ]; then
        echo -e "${YELLOW}⚠️  Dependencies are missing from your virtual environment${NC}"
        echo ""
        read -p "$(echo -e ${YELLOW}Would you like to install all dependencies from pyproject.toml? [y/N]: ${NC})" -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}📥 Installing dependencies from pyproject.toml...${NC}"
            "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet
            # Install dependencies directly without editable mode to avoid package discovery issues
            "$VENV_PYTHON" << 'INSTALL_EOF'
import sys
import re
from pathlib import Path
import subprocess

def extract_dependencies():
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return []
    
    try:
        try:
            import tomllib
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)
        except ImportError:
            try:
                import tomli as tomllib
                with open(pyproject_path, 'rb') as f:
                    data = tomllib.load(f)
            except ImportError:
                raise ImportError("No TOML parser available")
        
        return data.get('project', {}).get('dependencies', [])
    except (ImportError, KeyError):
        # Fallback: manual parsing
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        dependencies = []
        in_dependencies = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('dependencies = ['):
                in_dependencies = True
                continue
            elif in_dependencies:
                if line.startswith(']'):
                    break
                match = re.search(r'["\']([^"\']+)["\']', line)
                if match:
                    dependencies.append(match.group(1))
        return dependencies

deps = extract_dependencies()
if deps:
    print(f"Installing {len(deps)} packages...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + deps)
INSTALL_EOF
            echo -e "${GREEN}✅ Dependencies installed${NC}"
        else
            echo -e "${YELLOW}⚠️  Skipping installation. The server may not work correctly without these packages.${NC}"
        fi
    else
        echo -e "${GREEN}✅ Dependencies found in virtual environment${NC}"
    fi
    
    PYTHON_ENV_CMD="$VENV_PYTHON"
fi

# Check database connection
echo ""
echo -e "${BLUE}🔍 Checking database connection...${NC}"

# Check if ~/dot_env.txt exists
if [ -f ~/dot_env.txt ]; then
    echo -e "${GREEN}✅ Found shared database configuration (~/dot_env.txt)${NC}"
fi

# Try to check PostgreSQL connection
if $PYTHON_ENV_CMD scripts/setup_postgres.py --check-only 2>/dev/null; then
    echo -e "${GREEN}✅ Database connection successful${NC}"
else
    # Check if it's using SQLite (which is fine for local dev)
    if $PYTHON_ENV_CMD -c "from app.core.config import settings; exit(0 if 'sqlite' in settings.database_url else 1)" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Using local SQLite database (not shared)${NC}"
        echo -e "${YELLOW}   To use shared database, run: python scripts/setup_shared_database.py${NC}"
    else
        echo -e "${YELLOW}⚠️  Database connection check failed${NC}"
        echo -e "${YELLOW}   Attempting to initialize database...${NC}"
        
        # Try to initialize database (will fail gracefully if already initialized)
        if $PYTHON_ENV_CMD scripts/setup_postgres.py 2>/dev/null; then
            echo -e "${GREEN}✅ Database initialized${NC}"
        else
            echo -e "${YELLOW}⚠️  Database initialization skipped (may already be set up)${NC}"
        fi
    fi
fi

# Start the server
echo ""
echo -e "${BLUE}🌐 Starting FastAPI server...${NC}"
echo -e "${GREEN}📍 Server will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}📚 API documentation at: http://localhost:8000/api/docs${NC}"
echo -e "${YELLOW}🛑 Press Ctrl+C to stop the server${NC}"
echo "=========================================="
echo ""

# Start uvicorn
exec $PYTHON_ENV_CMD -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

