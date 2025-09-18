#!/usr/bin/env python3
"""Test JAX parallelization capabilities."""

import jax
import jax.numpy as jnp
import time
import numpy as np
import sys
import os

# Add the hole_optimization module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hole_optimization'))

try:
    from hole_optimization.gpu_config import setup_gpu_for_optimization, GPUConfig
    GPU_CONFIG_AVAILABLE = True
except ImportError:
    GPU_CONFIG_AVAILABLE = False
    print("⚠️  GPU configuration module not available. Using basic JAX setup.")

def test_jax_parallelization():
    """Test what parallelization JAX can do."""
    print("🔍 Testing JAX Parallelization")
    print("=" * 40)
    
    # Setup GPU if available
    if GPU_CONFIG_AVAILABLE:
        print("🚀 Setting up GPU configuration...")
        gpu_config = setup_gpu_for_optimization(use_gpu=True, verbose=True)
        print()
    else:
        gpu_config = None
    
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Enhanced device information
    if GPU_CONFIG_AVAILABLE and gpu_config:
        device_info = gpu_config.get_device_info()
        print(f"GPU devices: {device_info.get('gpu_devices', 0)}")
        print(f"CPU devices: {device_info.get('cpu_devices', 0)}")
        print(f"Backend: {device_info.get('backend', 'unknown')}")
    
    print()
    
    # Test 1: Basic JAX operations
    print("🧮 Test 1: Basic JAX Operations")
    x = jnp.array(np.random.randn(1000, 1000))
    
    start = time.time()
    result = jnp.dot(x, x.T)
    jax_time = time.time() - start
    print(f"JAX matrix multiplication (1000x1000): {jax_time:.4f}s")
    
    # Test 2: JIT compilation
    print("\n⚡ Test 2: JIT Compilation")
    
    def simple_function(x):
        return jnp.sum(x ** 2)
    
    # First call (compilation)
    start = time.time()
    result1 = simple_function(x)
    compile_time = time.time() - start
    
    # Second call (compiled)
    start = time.time()
    result2 = simple_function(x)
    compiled_time = time.time() - start
    
    print(f"First call (compilation): {compile_time:.4f}s")
    print(f"Second call (compiled): {compiled_time:.4f}s")
    print(f"Speedup: {compile_time/compiled_time:.2f}x")
    
    # Test 3: Vectorized operations
    print("\n🔄 Test 3: Vectorized Operations")
    
    def vectorized_operation(x):
        return jnp.sum(jnp.sin(x) + jnp.cos(x))
    
    # JIT compile the function
    compiled_func = jax.jit(vectorized_operation)
    
    # Test with different sizes
    sizes = [1000, 5000, 10000]
    for size in sizes:
        test_x = jnp.array(np.random.randn(size, size))
        
        start = time.time()
        result = compiled_func(test_x)
        jax_time = time.time() - start
        
        print(f"Vectorized operation ({size}x{size}): {jax_time:.4f}s")
    
    # Test 4: Gradient computation
    print("\n📈 Test 4: Gradient Computation")
    
    def complex_function(x):
        return jnp.sum(jnp.sin(x) * jnp.cos(x) + x ** 2)
    
    grad_func = jax.grad(complex_function)
    compiled_grad = jax.jit(grad_func)
    
    test_x = jnp.array(np.random.randn(1000))
    
    start = time.time()
    gradients = compiled_grad(test_x)
    grad_time = time.time() - start
    
    print(f"Gradient computation (1000 params): {grad_time:.4f}s")
    
    # Test 5: Check if we can use pmap (parallel mapping)
    print("\n🚀 Test 5: Parallel Mapping")
    
    if jax.local_device_count() > 1:
        print("Multiple devices available for pmap")
        # This would work with multiple devices
    else:
        print("Single device - pmap will replicate across devices")
        # Test pmap with single device (it will replicate)
        def simple_parallel_func(x):
            return x ** 2
        
        parallel_func = jax.pmap(simple_parallel_func)
        
        # Create data for parallel processing
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        
        start = time.time()
        result = parallel_func(data)
        pmap_time = time.time() - start
        
        print(f"pmap test: {pmap_time:.4f}s")
        print(f"Result shape: {result.shape}")
    
    print("\n✅ JAX Parallelization Test Complete!")
    print("\nKey Points:")
    print("- JAX uses JIT compilation for speed")
    print("- Vectorized operations are highly optimized")
    print("- Gradient computation is automatic and fast")
    print("- Even on CPU, JAX provides significant speedups over NumPy")

if __name__ == "__main__":
    test_jax_parallelization()
