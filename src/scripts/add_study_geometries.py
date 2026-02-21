#!/usr/bin/env python3
"""
Add study geometries to the database.

Updates Control to β:95° and adds 12 study geometries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.connection import SessionLocal
from database.models import CrutchGeometry


def main():
    """Add study geometries."""
    print("Adding study geometries...")
    
    session = SessionLocal()
    
    try:
        # Update existing Control to have β:95°
        print("\n1. Updating Control geometry...")
        control = session.query(CrutchGeometry).filter(
            CrutchGeometry.name == "Control"
        ).first()
        
        if control:
            old_beta = control.beta
            control.beta = 95.0
            print(f"   Updated Control: β:{old_beta}° → β:95.0°")
        else:
            # Create new Control if it doesn't exist
            control = CrutchGeometry(
                name="Control",
                alpha=95.0,
                beta=95.0,
                gamma=0.0,
                delta=0.0
            )
            session.add(control)
            print(f"   Created Control: α:95°, β:95°, γ:0°")
        
        session.commit()
        
        # Define the parameter space for study geometries
        alphas = [85, 105]
        betas = [95, 125]
        gammas = [-9, 0, 9]
        
        print("\n2. Adding study geometries...")
        
        # Find next available G number
        all_geoms = session.query(CrutchGeometry).all()
        g_numbers = []
        for g in all_geoms:
            if g.name.startswith('G') and g.name[1:].isdigit():
                g_numbers.append(int(g.name[1:]))
        
        next_g = max(g_numbers) + 1 if g_numbers else 1
        
        created_count = 0
        skipped_count = 0
        
        for gamma in gammas:
            print(f"\n   Gamma {gamma}°:")
            for alpha in alphas:
                for beta in betas:
                    # Check if this geometry already exists
                    exists = session.query(CrutchGeometry).filter(
                        CrutchGeometry.alpha == float(alpha),
                        CrutchGeometry.beta == float(beta),
                        CrutchGeometry.gamma == float(gamma)
                    ).first()
                    
                    if not exists:
                        name = f"G{next_g}"
                        geom = CrutchGeometry(
                            name=name,
                            alpha=float(alpha),
                            beta=float(beta),
                            gamma=float(gamma),
                            delta=0.0
                        )
                        session.add(geom)
                        print(f"     ✓ {name}: α:{alpha}°, β:{beta}°, γ:{gamma}°")
                        next_g += 1
                        created_count += 1
                    else:
                        print(f"     ⊙ α:{alpha}°, β:{beta}°, γ:{gamma}° (already exists)")
                        skipped_count += 1
        
        session.commit()
        
        # Verify
        study_geoms = session.query(CrutchGeometry).filter(
            CrutchGeometry.gamma.in_([-9, 0, 9]),
            CrutchGeometry.alpha.in_([85, 105]),
            CrutchGeometry.beta.in_([95, 125])
        ).all()
        
        print(f"\n✅ Success!")
        print(f"\nCreated: {created_count} geometries")
        print(f"Skipped: {skipped_count} (already existed)")
        print(f"\nStudy geometries now available: {len(study_geoms)}")
        print(f"  • Control: α:95°, β:95°, γ:0°")
        print(f"  • 12 geometries: α∈{{85,105}}, β∈{{95,125}}, γ∈{{-9,0,9}}")
        print(f"  • Total: 13 trials (baseline + 4 + 4 + 4 + baseline)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()

