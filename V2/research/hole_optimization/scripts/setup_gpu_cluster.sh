#!/bin/bash
# Setup script for GPU cluster deployment

set -e  # Exit on any error

echo "🚀 Setting up Personalising Crutches for GPU Cluster"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the hole_optimization directory."
    exit 1
fi

# Detect CUDA version
echo "🔍 Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "Found CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvidia-smi not found. Assuming CUDA 12.x"
    CUDA_VERSION="12"
fi

# Set up environment variables
echo "🔧 Setting up environment variables..."
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install JAX with GPU support based on CUDA version
echo "🔧 Installing JAX with GPU support..."
if [[ "$CUDA_VERSION" == "12"* ]]; then
    echo "Installing JAX for CUDA 12.x..."
    pip install "jax[cuda12_pip]>=0.4.0" jaxlib
elif [[ "$CUDA_VERSION" == "11"* ]]; then
    echo "Installing JAX for CUDA 11.x..."
    pip install "jax[cuda11_pip]>=0.4.0" jaxlib
else
    echo "Installing JAX for CUDA 11.x (fallback)..."
    pip install "jax[cuda11_pip]>=0.4.0" jaxlib
fi

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Test GPU setup
echo "🧪 Testing GPU setup..."
python3 -c "
import jax
import jax.numpy as jnp
print(f'JAX version: {jax.__version__}')
print(f'Available devices: {jax.devices()}')
print(f'Default backend: {jax.default_backend()}')

# Test GPU computation
if any(d.device_kind == 'gpu' for d in jax.devices()):
    print('✅ GPU is available!')
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.dot(x, x.T)
    print(f'GPU computation test: {y}')
else:
    print('⚠️  GPU not available, using CPU')
"

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test GPU functionality, run:"
echo "  python3 ../test_jax_parallelization.py"
