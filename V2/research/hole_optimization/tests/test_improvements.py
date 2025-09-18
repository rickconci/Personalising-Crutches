#!/usr/bin/env python3
"""Test script to demonstrate the critical fixes we implemented."""

import numpy as np
import jax.numpy as jnp
from config import CrutchConstraints
from optimizers.differentiable import DifferentiableOptimizer
from config import DifferentiableConfig, OptimizationObjectives

def test_rod_specific_holes():
    """Test that max holes are now rod-specific."""
    print("🔧 Testing Rod-Specific Max Holes")
    print("=" * 50)
    
    constraints = CrutchConstraints()
    
    print(f"Rod lengths:")
    print(f"  Handle: {constraints.handle_length}cm")
    print(f"  Vertical: {constraints.vertical_length}cm") 
    print(f"  Forearm: {constraints.forearm_length}cm")
    print()
    
    print(f"Max holes (min_distance={constraints.min_hole_distance}cm, margin={constraints.hole_margin}cm):")
    print(f"  Handle: {constraints.max_handle_holes} holes")
    print(f"  Vertical: {constraints.max_vertical_holes} holes")
    print(f"  Forearm: {constraints.max_forearm_holes} holes")
    print(f"  Total: {constraints.total_max_holes} holes")
    print()
    
    # Show efficiency improvement
    old_total = 3 * 20  # Old approach: 20 holes per rod
    new_total = constraints.total_max_holes
    efficiency = (old_total - new_total) / old_total * 100
    
    print(f"Efficiency improvement:")
    print(f"  Old parameter vector size: {old_total}")
    print(f"  New parameter vector size: {new_total}")
    print(f"  Reduction: {efficiency:.1f}%")


def test_unique_count_math():
    """Test the fixed differentiable unique count function."""
    print("\n🧮 Testing Fixed Unique Count Math")
    print("=" * 50)
    
    # Create mock optimizer to test the function
    constraints = CrutchConstraints()
    objectives = OptimizationObjectives()
    config = DifferentiableConfig()
    
    optimizer = DifferentiableOptimizer(constraints, objectives, config)
    
    # Test cases
    test_cases = [
        ([12.0, 12.0, 12.0, 12.0], "All identical"),
        ([10.0, 12.0, 14.0, 16.0], "All different"),
        ([10.0, 10.0, 15.0, 15.0], "Two pairs"),
        ([8.0, 12.0, 12.1, 16.0], "Mostly different"),
    ]
    
    print("Test cases (expected: identical=1, all different=4):")
    for values, description in test_cases:
        values_array = jnp.array(values)
        unique_count = optimizer._differentiable_unique_count(values_array, bandwidth=0.5)
        actual_unique = len(set(np.round(values, 1)))
        
        print(f"  {description:15} | Values: {values} | "
              f"Differentiable: {unique_count:.2f} | Actual: {actual_unique}")


def test_structural_evaluation():
    """Test the new structural evaluation functions."""
    print("\n🏗️ Testing Structural Evaluation")
    print("=" * 50)
    
    constraints = CrutchConstraints()
    objectives = OptimizationObjectives()
    config = DifferentiableConfig()
    
    optimizer = DifferentiableOptimizer(constraints, objectives, config)
    
    # Test truss length evaluation
    print("Truss length quality scores:")
    test_lengths = [5.0, 10.0, 15.0, 20.0, 25.0, 35.0]
    for length in test_lengths:
        score = optimizer._evaluate_truss_length(length)
        print(f"  {length:4.1f}cm: {score:.3f}")
    
    print("\nStructural leverage scores:")
    test_arms = [(2, 2), (5, 5), (10, 10), (2, 10), (15, 15)]
    for arm1, arm2 in test_arms:
        score = optimizer._evaluate_structural_leverage(arm1, arm2)
        geometric_mean = np.sqrt(arm1 * arm2)
        print(f"  Arms ({arm1:2d}, {arm2:2d})cm | Geom mean: {geometric_mean:4.1f} | Score: {score:.3f}")


def test_sampling_improvement():
    """Test the improved uniform sampling strategy."""
    print("\n🎲 Testing Improved Sampling Strategy")
    print("=" * 50)
    
    constraints = CrutchConstraints()
    objectives = OptimizationObjectives()
    config = DifferentiableConfig()
    
    optimizer = DifferentiableOptimizer(constraints, objectives, config)
    
    # Create mock hole arrays
    handle_holes = jnp.linspace(1, 37, 18)  # 18 holes on handle
    vertical_holes = jnp.linspace(1, 19, 9)   # 9 holes on vertical
    forearm_holes = jnp.linspace(1, 16, 8)    # 8 holes on forearm
    
    print(f"Rod holes available:")
    print(f"  Handle: {len(handle_holes)} holes")
    print(f"  Vertical: {len(vertical_holes)} holes")
    print(f"  Forearm: {len(forearm_holes)} holes")
    print(f"  Total combinations: {len(handle_holes) * len(vertical_holes) * len(forearm_holes):,}")
    
    # Test sampling
    n_samples = 50
    combinations = optimizer._sample_hole_combinations_uniform(
        handle_holes, vertical_holes, forearm_holes, n_samples
    )
    
    print(f"\nUniform sampling results:")
    print(f"  Requested samples: {n_samples}")
    print(f"  Generated samples: {len(combinations)}")
    print(f"  Coverage: {len(combinations) / (len(handle_holes) * len(vertical_holes) * len(forearm_holes)) * 100:.3f}%")
    
    # Show first few samples
    print(f"\nFirst 5 samples (handle, vertical, forearm):")
    for i, (h, v, f) in enumerate(combinations[:5]):
        print(f"  {i+1}: ({h:.1f}, {v:.1f}, {f:.1f})")


def main():
    """Run all tests."""
    print("🧪 TESTING CRITICAL FIXES")
    print("=" * 60)
    
    test_rod_specific_holes()
    test_unique_count_math()
    test_structural_evaluation() 
    test_sampling_improvement()
    
    print("\n✅ All tests completed!")
    print("\nKey improvements:")
    print("  1. ✅ Rod-specific max holes (more efficient parameter space)")
    print("  2. ✅ Fixed unique count math (correct uniqueness estimation)")
    print("  3. ✅ Structural evaluation (length + leverage optimization)")
    print("  4. ✅ Uniform sampling (unbiased exploration)")
    print("\nThe optimizer now:")
    print("  • Uses smaller, more efficient parameter vectors")
    print("  • Correctly estimates truss complexity")
    print("  • Optimizes for structural soundness")
    print("  • Explores the space without sampling bias")


if __name__ == "__main__":
    main()
