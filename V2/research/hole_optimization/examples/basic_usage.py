#!/usr/bin/env python3
"""Example usage of the hole optimization framework."""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run example optimization."""
    try:
        from hole_optimization import (
            ConfigManager, 
            create_optimizer, 
            plot_optimization_results
        )
        from hole_optimization.config import OptimizerType
        
        logger.info("🚀 Starting hole optimization example")
        
        # Create output directory
        output_dir = Path("example_results")
        output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        config_file = Path("configs/default_differentiable.yaml")
        
        if config_file.exists():
            logger.info(f"📄 Loading config from {config_file}")
            config = ConfigManager.load_config(config_file)
        else:
            logger.info("📄 Using default configuration")
            from hole_optimization.config import ExperimentConfig
            config = ExperimentConfig()
            config.optimizer_type = OptimizerType.DIFFERENTIABLE
        
        # Print configuration summary
        logger.info("⚙️ Configuration Summary:")
        logger.info(f"  • Optimizer: {config.optimizer_type.value}")
        logger.info(f"  • Vertical length: {config.constraints.vertical_length}cm")
        logger.info(f"  • Handle length: {config.constraints.handle_length}cm")
        logger.info(f"  • Forearm length: {config.constraints.forearm_length}cm")
        logger.info(f"  • Alpha range: {config.constraints.alpha_min}° - {config.constraints.alpha_max}°")
        logger.info(f"  • Beta range: {config.constraints.beta_min}° - {config.constraints.beta_max}°")
        
        # Create optimizer
        logger.info("🔧 Creating optimizer...")
        optimizer = create_optimizer(
            config.optimizer_type,
            config.constraints,
            config.objectives,
            config.differentiable,  # Using differentiable config
            config.random_seed
        )
        
        # Run optimization
        logger.info("🎯 Running optimization...")
        result = optimizer.optimize()
        
        # Print results
        logger.info("✅ Optimization completed!")
        logger.info("📊 Results Summary:")
        logger.info(f"  • Vocabulary size: {result.best_metrics.vocabulary_size} geometries")
        logger.info(f"  • Unique trusses: {result.best_metrics.unique_truss_count}")
        logger.info(f"  • Total holes: {result.best_hole_layout.total_holes}")
        logger.info(f"  • Runtime: {result.total_time:.2f}s")
        logger.info(f"  • Iterations: {result.iterations}")
        logger.info(f"  • Converged: {'Yes' if result.converged else 'No'}")
        
        # Hole distribution
        logger.info("🕳️ Hole Distribution:")
        logger.info(f"  • Handle: {len(result.best_hole_layout.handle)} holes")
        logger.info(f"  • Vertical: {len(result.best_hole_layout.vertical)} holes")
        logger.info(f"  • Forearm: {len(result.best_hole_layout.forearm)} holes")
        
        # Save results
        logger.info("💾 Saving results...")
        
        # Create visualizations
        try:
            plot_optimization_results(result, output_dir)
            logger.info(f"📈 Visualizations saved to {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
        
        # Print some example geometries
        if result.best_geometries:
            logger.info("🎨 Sample Geometries:")
            for i, geom in enumerate(result.best_geometries[:5]):
                logger.info(f"  {i+1}. α={geom.alpha:.1f}°, β={geom.beta:.1f}°, "
                          f"trusses=[{geom.truss_1:.1f}, {geom.truss_2:.1f}, {geom.truss_3:.1f}]cm")
        
        logger.info(f"🎉 Example completed! Check {output_dir} for results.")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure to install dependencies: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
