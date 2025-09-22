"""GPU configuration utilities for JAX optimization."""

import os
import logging
from typing import Optional, List, Dict, Any
import warnings

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    warnings.warn("JAX not available. GPU configuration will be limited.")


class GPUConfig:
    """Configuration manager for GPU usage in JAX optimization."""
    
    def __init__(
        self,
        use_gpu: bool = True,
        gpu_memory_fraction: float = 0.8,
        allow_gpu_growth: bool = True,
        prefer_gpu: bool = True,
        fallback_to_cpu: bool = True
    ):
        """Initialize GPU configuration.
        
        Args:
            use_gpu: Whether to attempt GPU usage
            gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
            allow_gpu_growth: Allow GPU memory to grow dynamically
            prefer_gpu: Prefer GPU over CPU when available
            fallback_to_cpu: Fall back to CPU if GPU fails
        """
        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.allow_gpu_growth = allow_gpu_growth
        self.prefer_gpu = prefer_gpu
        self.fallback_to_cpu = fallback_to_cpu
        
        self.logger = logging.getLogger(f"{__name__}.GPUConfig")
        self._devices = None
        self._backend = None
        
    def setup_jax_gpu(self) -> bool:
        """Setup JAX for GPU usage.
        
        Returns:
            True if GPU setup successful, False otherwise
        """
        if not JAX_AVAILABLE:
            self.logger.warning("JAX not available. Cannot configure GPU.")
            return False
            
        if not self.use_gpu:
            self.logger.info("GPU usage disabled by configuration. Using CPU.")
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            self._backend = 'cpu'
            return True
            
        try:
            # Set environment variables for GPU configuration
            self._set_gpu_environment_vars()
            
            # Check available devices
            devices = jax.devices()
            self._devices = devices
            
            # Check if we have GPU devices
            gpu_devices = [d for d in devices if d.device_kind == 'gpu']
            
            if gpu_devices:
                self.logger.info(f"✅ Found {len(gpu_devices)} GPU device(s): {gpu_devices}")
                self._backend = 'gpu'
                return True
            else:
                self.logger.warning("No GPU devices found.")
                if self.fallback_to_cpu:
                    self.logger.info("Falling back to CPU.")
                    self._backend = 'cpu'
                    return True
                else:
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to setup GPU: {e}")
            if self.fallback_to_cpu:
                self.logger.info("Falling back to CPU due to GPU setup failure.")
                self._backend = 'cpu'
                return True
            return False
    
    def _set_gpu_environment_vars(self) -> None:
        """Set environment variables for GPU configuration."""
        # CUDA memory management
        if self.gpu_memory_fraction < 1.0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU by default
            
        # JAX specific settings
        if self.allow_gpu_growth:
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        else:
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
            
        # Memory fraction (approximate)
        if self.gpu_memory_fraction < 1.0:
            # JAX doesn't have direct memory fraction control like TensorFlow
            # but we can set XLA flags
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.gpu_memory_fraction)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices.
        
        Returns:
            Dictionary with device information
        """
        if not JAX_AVAILABLE:
            return {"error": "JAX not available"}
            
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        cpu_devices = [d for d in devices if d.device_kind == 'cpu']
        
        return {
            "total_devices": len(devices),
            "gpu_devices": len(gpu_devices),
            "cpu_devices": len(cpu_devices),
            "devices": [str(d) for d in devices],
            "gpu_device_names": [str(d) for d in gpu_devices],
            "backend": jax.default_backend(),
            "local_device_count": jax.local_device_count(),
            "jax_version": jax.__version__
        }
    
    def test_gpu_performance(self, matrix_size: int = 1000) -> Dict[str, float]:
        """Test GPU performance with matrix operations.
        
        Args:
            matrix_size: Size of test matrices
            
        Returns:
            Dictionary with performance metrics
        """
        if not JAX_AVAILABLE:
            return {"error": "JAX not available"}
            
        import time
        
        # Create test data
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (matrix_size, matrix_size))
        
        # Test matrix multiplication
        def matmul_test(x):
            return jnp.dot(x, x.T)
        
        # JIT compile
        compiled_matmul = jax.jit(matmul_test)
        
        # Warmup
        _ = compiled_matmul(x)
        
        # Time the operation
        start_time = time.time()
        result = compiled_matmul(x)
        end_time = time.time()
        
        # Ensure computation is done
        result.block_until_ready()
        
        execution_time = end_time - start_time
        flops = 2 * matrix_size ** 3  # Approximate FLOPS for matrix multiplication
        gflops = flops / (execution_time * 1e9)
        
        return {
            "execution_time": execution_time,
            "matrix_size": matrix_size,
            "gflops": gflops,
            "device": str(result.device()),
            "backend": jax.default_backend()
        }
    
    @property
    def devices(self) -> Optional[List]:
        """Get available devices."""
        return self._devices
    
    @property
    def backend(self) -> Optional[str]:
        """Get current backend."""
        return self._backend
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and working."""
        if not JAX_AVAILABLE:
            return False
        try:
            devices = jax.devices()
            return any(d.device_kind == 'gpu' for d in devices)
        except Exception:
            return False


def setup_gpu_for_optimization(
    use_gpu: bool = True,
    gpu_memory_fraction: float = 0.8,
    verbose: bool = True
) -> GPUConfig:
    """Convenience function to setup GPU for optimization.
    
    Args:
        use_gpu: Whether to use GPU
        gpu_memory_fraction: Fraction of GPU memory to use
        verbose: Whether to print setup information
        
    Returns:
        Configured GPUConfig instance
    """
    config = GPUConfig(
        use_gpu=use_gpu,
        gpu_memory_fraction=gpu_memory_fraction
    )
    
    success = config.setup_jax_gpu()
    
    if verbose:
        device_info = config.get_device_info()
        print("🔧 GPU Configuration Setup")
        print("=" * 40)
        print(f"GPU Available: {config.is_gpu_available()}")
        print(f"Backend: {device_info.get('backend', 'unknown')}")
        print(f"Devices: {device_info.get('total_devices', 0)} total")
        print(f"GPU Devices: {device_info.get('gpu_devices', 0)}")
        print(f"CPU Devices: {device_info.get('cpu_devices', 0)}")
        
        if success and config.is_gpu_available():
            print("✅ GPU setup successful!")
            # Run performance test
            perf = config.test_gpu_performance()
            print(f"Performance Test: {perf['gflops']:.2f} GFLOPS")
        else:
            print("⚠️  Using CPU (GPU not available or failed)")
    
    return config


if __name__ == "__main__":
    # Test GPU configuration
    gpu_config = setup_gpu_for_optimization(verbose=True)
    
    if gpu_config.is_gpu_available():
        print("\n🚀 Running GPU performance test...")
        perf = gpu_config.test_gpu_performance(matrix_size=2000)
        print(f"Matrix multiplication (2000x2000): {perf['execution_time']:.4f}s")
        print(f"Performance: {perf['gflops']:.2f} GFLOPS")
    else:
        print("\n💻 Running CPU performance test...")
        perf = gpu_config.test_gpu_performance(matrix_size=1000)
        print(f"Matrix multiplication (1000x1000): {perf['execution_time']:.4f}s")
