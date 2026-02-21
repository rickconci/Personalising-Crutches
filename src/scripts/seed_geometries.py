#!/usr/bin/env python3
"""
Seed the database with the correct geometry configurations.

Creates:
- Control (Baseline): α:95°, β:95°, γ:0°
- 12 regular geometries: α∈{85, 105}, β∈{95, 125}, γ∈{-9, 0, 9}
  Total: 4 geometries per gamma × 3 gammas = 12 geometries
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import SessionLocal
from database.models import CrutchGeometry
from sqlalchemy import delete


def main():
    """Seed geometries in the database."""
    print("Starting geometry seeding...")
    print("This will ADD new geometries without removing existing ones.\n")
    
    session = SessionLocal()
    
    try:
        # Check existing geometries
        existing_count = session.query(CrutchGeometry).count()
        print(f"Current geometries in database: {existing_count}")
        
        # Create Control/Baseline geometry (check if it exists first)
        print("\nCreating Control geometry...")
        control_exists = session.query(CrutchGeometry).filter(
            CrutchGeometry.alpha == 95.0,
            CrutchGeometry.beta == 95.0,
            CrutchGeometry.gamma == 0.0,
            CrutchGeometry.name.like("Control%")
        ).first()
        
        if not control_exists:
            control = CrutchGeometry(
                name="Control",
                alpha=95.0,
                beta=95.0,
                gamma=0.0,
                delta=0.0
            )
            session.add(control)
            print(f"  ✓ Control: α:95°, β:95°, γ:0° (created)")
        else:
            print(f"  ⊙ Control already exists (skipped)")
        
        # Define the parameter space
        alphas = [85, 105]
        betas = [95, 125]
        gammas = [-9, 0, 9]
        
        print("\nCreating regular geometries...")
        geometry_counter = 1
        
        # Create geometries for each combination (check if they exist first)
        for gamma in gammas:
            print(f"\n  Gamma {gamma}°:")
            for alpha in alphas:
                for beta in betas:
                    # Check if this geometry already exists
                    exists = session.query(CrutchGeometry).filter(
                        CrutchGeometry.alpha == float(alpha),
                        CrutchGeometry.beta == float(beta),
                        CrutchGeometry.gamma == float(gamma)
                    ).first()
                    
                    if not exists:
                        # Find next available name
                        existing_names = [g.name for g in session.query(CrutchGeometry).all()]
                        while f"G{geometry_counter}" in existing_names:
                            geometry_counter += 1
                        
                        name = f"G{geometry_counter}"
                        geom = CrutchGeometry(
                            name=name,
                            alpha=float(alpha),
                            beta=float(beta),
                            gamma=float(gamma),
                            delta=0.0
                        )
                        session.add(geom)
                        print(f"    ✓ {name}: α:{alpha}°, β:{beta}°, γ:{gamma}° (created)")
                        geometry_counter += 1
                    else:
                        print(f"    ⊙ α:{alpha}°, β:{beta}°, γ:{gamma}° already exists (skipped)")
        
        # Commit all changes
        session.commit()
        
        # Verify
        total = session.query(CrutchGeometry).count()
        new_count = total - existing_count
        
        # Count study geometries (the ones we care about)
        study_geometries = session.query(CrutchGeometry).filter(
            CrutchGeometry.gamma.in_([-9, 0, 9]),
            CrutchGeometry.alpha.in_([85, 105]),
            CrutchGeometry.beta.in_([95, 125])
        ).count()
        
        control_count = session.query(CrutchGeometry).filter(
            CrutchGeometry.alpha == 95.0,
            CrutchGeometry.beta == 95.0,
            CrutchGeometry.gamma == 0.0
        ).count()
        
        print(f"\n✅ Success!")
        print(f"\nAdded {new_count} new geometries")
        print(f"Total geometries in database: {total}")
        print(f"\nStudy geometries available:")
        print(f"  • Control/Baseline: {control_count}")
        print(f"  • Regular geometries (α∈{{85,105}}, β∈{{95,125}}, γ∈{{-9,0,9}}): {study_geometries}")
        print(f"  • Study total: {control_count + study_geometries}")
        print(f"\nNote: The systematic mode frontend will only show the study geometries.")
        
    except Exception as e:
        print(f"\n❌ Error seeding geometries: {e}")
        session.rollback()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()

