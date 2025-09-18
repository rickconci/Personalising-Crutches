#!/usr/bin/env python3
"""
Demonstration of the Hole Optimization Framework

This script showcases the key features of our professional hole optimization system:
1. Configuration management
2. Multiple optimization algorithms  
3. Benchmarking capabilities
4. Visualization tools

Run with: python demo.py
"""

import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_optimization():
    """Demonstrate basic optimization workflow."""
    logger.info("🚀 DEMO 1: Basic Optimization Workflow")
    logger.info("=" * 50)
    
    try:
        from hole_optimization.config import ExperimentConfig, OptimizerType
        from hole_optimization.optimizers.factory import create_optimizer
        from hole_optimization.geometry import create_uniform_holes
        
        # Create default configuration
        config = ExperimentConfig()
        config.optimizer_type = OptimizerType.DIFFERENTIABLE
        config.differentiable.max_iterations = 100  # Quick demo
        
        logger.info(f"📋 Configuration:")
        logger.info(f"  • Optimizer: {config.optimizer_type.value}")
        logger.info(f"  • Max iterations: {config.differentiable.max_iterations}")
        logger.info(f"  • Vocabulary weight: {config.objectives.vocabulary_weight}")
        logger.info(f"  • Truss complexity weight: {config.objectives.truss_complexity_weight}")
        
        # Create optimizer
        optimizer = create_optimizer(
            config.optimizer_type,
            config.constraints,
            config.objectives,
            config.differentiable,
            config.random_seed
        )
        
        # Create initial hole layout
        initial_layout = create_uniform_holes(config.constraints)
        logger.info(f"📍 Initial layout: {initial_layout.total_holes} holes total")
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(initial_layout=initial_layout)
        end_time = time.time()
        
        # Display results
        logger.info("✅ Optimization Results:")
        logger.info(f"  • Runtime: {end_time - start_time:.2f}s")
        logger.info(f"  • Iterations: {result.iterations}")
        logger.info(f"  • Converged: {'Yes' if result.converged else 'No'}")
        logger.info(f"  • Vocabulary size: {result.best_metrics.vocabulary_size}")
        logger.info(f"  • Unique trusses: {result.best_metrics.unique_truss_count}")
        logger.info(f"  • Pareto efficiency: {result.best_metrics.pareto_efficiency:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Demo 1 failed: {e}")
        return None


def demo_configuration_system():
    """Demonstrate configuration management."""
    logger.info("\n🔧 DEMO 2: Configuration System")
    logger.info("=" * 50)
    
    try:
        from hole_optimization.config import (
            ConfigManager, ExperimentConfig, OptimizerType,
            CrutchConstraints, OptimizationObjectives
        )
        
        # Create custom configuration
        constraints = CrutchConstraints(
            vertical_length=25.0,  # Longer vertical rod
            handle_length=40.0,    # Longer handle
            alpha_min=80.0,        # Wider angle range
            alpha_max=120.0,
            min_hole_distance=1.5  # Closer hole spacing
        )
        
        objectives = OptimizationObjectives(
            vocabulary_weight=2.0,        # Prioritize vocabulary more
            truss_complexity_weight=0.3,  # Less concern about complexity
            max_unique_trusses=20         # Allow more unique trusses
        )
        
        config = ExperimentConfig(
            optimizer_type=OptimizerType.DIFFERENTIABLE,
            constraints=constraints,
            objectives=objectives,
            random_seed=123
        )
        
        logger.info("📝 Custom Configuration Created:")
        logger.info(f"  • Vertical length: {constraints.vertical_length}cm")
        logger.info(f"  • Handle length: {constraints.handle_length}cm") 
        logger.info(f"  • Alpha range: {constraints.alpha_min}° - {constraints.alpha_max}°")
        logger.info(f"  • Vocabulary weight: {objectives.vocabulary_weight}")
        logger.info(f"  • Max unique trusses: {objectives.max_unique_trusses}")
        
        # Save configuration
        config_dir = Path("demo_configs")
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "custom_config.yaml"
        
        try:
            ConfigManager.save_config(config, config_file)
            logger.info(f"💾 Configuration saved to: {config_file}")
            
            # Load it back
            loaded_config = ConfigManager.load_config(config_file)
            logger.info(f"📂 Configuration loaded successfully")
            logger.info(f"  • Loaded optimizer: {loaded_config.optimizer_type.value}")
            
        except ImportError as e:
            logger.warning(f"⚠️ YAML not available: {e}")
            logger.info("Install PyYAML for full config functionality: pip install PyYAML")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Demo 2 failed: {e}")
        return None


def demo_geometry_analysis():
    """Demonstrate geometry analysis capabilities."""
    logger.info("\n🎨 DEMO 3: Geometry Analysis")
    logger.info("=" * 50)
    
    try:
        from hole_optimization.geometry import CrutchGeometry, create_uniform_holes
        from hole_optimization.config import CrutchConstraints
        
        # Create geometry calculator
        constraints = CrutchConstraints()
        geometry_calc = CrutchGeometry(constraints)
        
        # Create sample hole layout
        layout = create_uniform_holes(constraints)
        
        logger.info("🕳️ Sample Hole Layout:")
        logger.info(f"  • Handle holes: {len(layout.handle)} positions")
        logger.info(f"  • Vertical holes: {len(layout.vertical)} positions") 
        logger.info(f"  • Forearm holes: {len(layout.forearm)} positions")
        logger.info(f"  • Total holes: {layout.total_holes}")
        
        # Calculate some example geometries
        logger.info("\n📐 Example Truss Calculations:")
        
        # Example 1: Truss 1 calculation
        alpha = 95.0  # degrees
        v_down = 5.0  # cm down from vertical pivot
        h_abs = 10.0  # cm from back of handle
        
        truss1_length = geometry_calc.truss1_length(alpha, v_down, h_abs)
        logger.info(f"  • Truss 1: α={alpha}°, v={v_down}cm, h={h_abs}cm → L={truss1_length:.2f}cm")
        
        # Example 2: Solve alpha from truss length
        target_truss = 12.0  # cm
        solved_alpha = geometry_calc.solve_alpha_from_truss1(v_down, h_abs, target_truss)
        logger.info(f"  • Solve α: target L={target_truss}cm → α={solved_alpha:.1f}°")
        
        # Enumerate some geometries
        sample_trusses = (12.0, 12.0, 12.0)  # T1, T2, T3
        geometries = geometry_calc.enumerate_geometries(layout, sample_trusses, length_tolerance=0.5)
        
        logger.info(f"\n🎯 Found {len(geometries)} valid geometries with target trusses {sample_trusses}")
        
        if geometries:
            logger.info("📊 Sample Geometries:")
            for i, geom in enumerate(geometries[:3]):
                logger.info(f"  {i+1}. α={geom.alpha:.1f}°, β={geom.beta:.1f}°, "
                          f"trusses=[{geom.truss_1:.1f}, {geom.truss_2:.1f}, {geom.truss_3:.1f}]")
        
        return geometries
        
    except Exception as e:
        logger.error(f"❌ Demo 3 failed: {e}")
        return None


def demo_multi_objective_analysis():
    """Demonstrate multi-objective optimization analysis."""
    logger.info("\n🎯 DEMO 4: Multi-Objective Analysis")
    logger.info("=" * 50)
    
    try:
        from hole_optimization.config import OptimizationObjectives
        
        # Define different objective weightings
        scenarios = [
            ("Vocabulary Priority", OptimizationObjectives(
                vocabulary_weight=2.0, truss_complexity_weight=0.2
            )),
            ("Balanced", OptimizationObjectives(
                vocabulary_weight=1.0, truss_complexity_weight=1.0
            )),
            ("Simplicity Priority", OptimizationObjectives(
                vocabulary_weight=0.3, truss_complexity_weight=2.0
            ))
        ]
        
        logger.info("🔄 Multi-Objective Scenarios:")
        
        for name, objectives in scenarios:
            logger.info(f"\n📋 {name}:")
            logger.info(f"  • Vocabulary weight: {objectives.vocabulary_weight}")
            logger.info(f"  • Truss complexity weight: {objectives.truss_complexity_weight}")
            logger.info(f"  • Trade-off ratio: {objectives.vocabulary_weight/objectives.truss_complexity_weight:.1f}:1")
            
            # Calculate example objective values
            vocab_size = 75
            truss_count = 12
            
            objective_value = (-vocab_size * objectives.vocabulary_weight + 
                             truss_count * objectives.truss_complexity_weight)
            
            logger.info(f"  • Example: {vocab_size} vocab, {truss_count} trusses → objective = {objective_value:.1f}")
        
        logger.info("\n💡 Insights:")
        logger.info("  • Vocabulary Priority: Maximizes design flexibility")
        logger.info("  • Simplicity Priority: Minimizes manufacturing complexity") 
        logger.info("  • Balanced: Good compromise for most applications")
        
        return scenarios
        
    except Exception as e:
        logger.error(f"❌ Demo 4 failed: {e}")
        return None


def main():
    """Run all demonstrations."""
    logger.info("🎪 HOLE OPTIMIZATION FRAMEWORK DEMONSTRATION")
    logger.info("=" * 60)
    logger.info("This demo showcases our professional optimization system")
    logger.info("for crutch hole placement optimization.")
    logger.info("")
    
    # Run demonstrations
    results = {}
    
    # Demo 1: Basic optimization
    results['basic'] = demo_basic_optimization()
    
    # Demo 2: Configuration system  
    results['config'] = demo_configuration_system()
    
    # Demo 3: Geometry analysis
    results['geometry'] = demo_geometry_analysis()
    
    # Demo 4: Multi-objective analysis
    results['multi_objective'] = demo_multi_objective_analysis()
    
    # Final summary
    logger.info("\n🎉 DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    
    success_count = sum(1 for result in results.values() if result is not None)
    logger.info(f"✅ Successful demos: {success_count}/{len(results)}")
    
    if success_count > 0:
        logger.info("\n🚀 Next Steps:")
        logger.info("  1. Install full dependencies: pip install -r requirements.txt")
        logger.info("  2. Run example.py for complete workflow")
        logger.info("  3. Try: python -m hole_optimization.main --help")
        logger.info("  4. Create custom configurations in configs/")
        logger.info("  5. Run benchmarks to compare algorithms")
        
        logger.info("\n📚 Key Features Demonstrated:")
        logger.info("  ✓ Professional configuration management")
        logger.info("  ✓ Modular optimizer architecture") 
        logger.info("  ✓ Comprehensive geometry calculations")
        logger.info("  ✓ Multi-objective optimization support")
        logger.info("  ✓ Extensible framework design")
    
    else:
        logger.warning("⚠️ Some demos failed. Check dependencies and error messages above.")
    
    logger.info("\n📖 See README.md for complete documentation!")


if __name__ == "__main__":
    main()
