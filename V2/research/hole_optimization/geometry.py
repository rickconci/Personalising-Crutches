"""Core geometry classes and functions for crutch optimization."""

from __future__ import annotations
from typing import List, Tuple, Optional
import jax.numpy as jnp
import jax
from pydantic import BaseModel, Field, field_validator
from .config import CrutchConstraints


class HoleLayout(BaseModel):
    """Represents hole positions on crutch rods."""
    handle: jnp.ndarray = Field(..., description="Absolute positions from back of handle (cm)")
    vertical: jnp.ndarray = Field(..., description="Distances down from vertical pivot (cm)")
    forearm: jnp.ndarray = Field(..., description="Distances up from forearm pivot (cm)")
    
    class Config:
        arbitrary_types_allowed = True  # For JAX arrays
    
    @field_validator('handle', 'vertical', 'forearm', mode='before')
    def ensure_jax_arrays(cls, v):
        """Ensure arrays are JAX arrays."""
        return jnp.asarray(v)
    
    @field_validator('handle', 'vertical', 'forearm')
    def validate_positive_positions(cls, v):
        """Validate that all positions are non-negative."""
        if jnp.any(v < 0):
            raise ValueError("All hole positions must be non-negative")
        return v
    
    @property
    def total_holes(self) -> int:
        """Total number of holes across all rods."""
        return len(self.handle) + len(self.vertical) + len(self.forearm)
    
    def to_flat_array(self) -> jnp.ndarray:
        """Convert to flat array for optimization."""
        return jnp.concatenate([self.handle, self.vertical, self.forearm])
    
    @classmethod
    def from_flat_array(
        cls, 
        flat_array: jnp.ndarray, 
        n_handle: int, 
        n_vertical: int, 
        n_forearm: int
    ) -> HoleLayout:
        """Create from flat array."""
        handle = flat_array[:n_handle]
        vertical = flat_array[n_handle:n_handle + n_vertical]
        forearm = flat_array[n_handle + n_vertical:n_handle + n_vertical + n_forearm]
        return cls(handle=handle, vertical=vertical, forearm=forearm)


class Geometry(BaseModel):
    """Represents a specific crutch geometry configuration."""
    # Angles
    alpha: float = Field(..., ge=0, le=180, description="Angle between vertical and handle (degrees)")
    beta: float = Field(..., ge=0, le=180, description="Angle between handle and forearm (degrees)")
    gamma: float = Field(..., description="Offset between pivots (cm)")
    
    # Hole selections
    t1_vertical: float = Field(..., ge=0, description="Vertical hole position for truss 1 (cm)")
    t1_handle: float = Field(..., ge=0, description="Handle hole position for truss 1 (cm)")
    t2_vertical: float = Field(..., ge=0, description="Vertical hole position for truss 2 (cm)")
    t2_handle: float = Field(..., ge=0, description="Handle hole position for truss 2 (cm)")
    t3_handle: float = Field(..., ge=0, description="Handle hole position for truss 3 (cm)")
    t3_forearm: float = Field(..., ge=0, description="Forearm hole position for truss 3 (cm)")
    
    # Truss lengths
    truss_1: float = Field(..., gt=0, description="Length of truss 1 (cm)")
    truss_2: float = Field(..., gt=0, description="Length of truss 2 (cm)")
    truss_3: float = Field(..., gt=0, description="Length of truss 3 (cm)")
    
    # Quality scores
    score_alpha: float = Field(..., ge=0, description="Angle quality score for truss 1")
    score_beta: float = Field(..., ge=0, description="Angle quality score for truss 3")
    
    @field_validator('alpha', 'beta')
    @classmethod
    def validate_reasonable_angles(cls, v, info):
        """Validate angles are in reasonable ranges."""
        if info.field_name == 'alpha' and not (60 <= v <= 150):
            raise ValueError(f"Alpha angle {v}° outside reasonable range [60°, 150°]")
        if info.field_name == 'beta' and not (60 <= v <= 180):
            raise ValueError(f"Beta angle {v}° outside reasonable range [60°, 180°]")
        return v
    
    def __hash__(self) -> int:
        """Hash based on rounded angle values for uniqueness checking."""
        return hash((round(self.alpha, 1), round(self.beta, 1)))
    
    def __eq__(self, other) -> bool:
        """Equality based on angles within tolerance."""
        if not isinstance(other, Geometry):
            return False
        return (abs(self.alpha - other.alpha) < 1.0 and 
                abs(self.beta - other.beta) < 1.0)


# Core geometric functions (JAX-compatible)

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between bounds."""
    return float(jnp.clip(x, lo, hi))


def law_of_cosines_length(r1: float, r2: float, theta_deg: float) -> float:
    """Calculate distance using law of cosines.
    
    Args:
        r1: First spoke length
        r2: Second spoke length
        theta_deg: Included angle in degrees
        
    Returns:
        Distance between spoke endpoints
    """
    theta_rad = jnp.deg2rad(theta_deg)
    return float(jnp.sqrt(jnp.maximum(
        0.0, r1*r1 + r2*r2 - 2.0*r1*r2*jnp.cos(theta_rad)
    )))


def solve_angle_from_lengths(a: float, b: float, c: float) -> float:
    """Solve included angle using law of cosines.
    
    Args:
        a: First side length
        b: Second side length  
        c: Opposite side length
        
    Returns:
        Included angle between a and b in degrees
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    
    cos_angle = clamp((a*a + b*b - c*c) / (2.0*a*b), -1.0, 1.0)
    return float(jnp.degrees(jnp.arccos(cos_angle)))


def angle_score_45deg(truss_length: float, spoke1: float, spoke2: float) -> float:
    """Score based on preference for 45° angles between truss and spokes.
    
    Args:
        truss_length: Length of truss
        spoke1: Length of first spoke
        spoke2: Length of second spoke
        
    Returns:
        Score (lower is better, 0 = both angles are 45°)
    """
    def angle_at_vertex(opposite: float, adjacent1: float, adjacent2: float) -> float:
        if adjacent1 == 0.0 or adjacent2 == 0.0:
            return 90.0
        cos_angle = clamp(
            (adjacent1*adjacent1 + adjacent2*adjacent2 - opposite*opposite) / 
            (2.0 * adjacent1 * adjacent2), -1.0, 1.0
        )
        return float(jnp.degrees(jnp.arccos(cos_angle)))
    
    # Angles at the two vertices where truss meets spokes
    angle1 = angle_at_vertex(spoke2, truss_length, spoke1)
    angle2 = angle_at_vertex(spoke1, truss_length, spoke2)
    
    return abs(angle1 - 45.0) + abs(angle2 - 45.0)


class CrutchGeometry:
    """Handles geometric calculations for crutch configurations."""
    
    def __init__(self, constraints: CrutchConstraints):
        """Initialize with physical constraints.
        
        Args:
            constraints: Physical constraints for the crutch
        """
        self.constraints = constraints
        
    def truss1_length(self, alpha: float, v_down: float, h_abs: float) -> float:
        """Calculate truss 1 length (vertical to handle, behind vertical pivot).
        
        Args:
            alpha: Angle between vertical and handle (degrees)
            v_down: Vertical hole distance down from pivot (cm)
            h_abs: Handle hole absolute position from back (cm)
            
        Returns:
            Truss length (cm)
        """
        handle_distance = abs(self.constraints.vertical_pivot_length - h_abs)
        return law_of_cosines_length(v_down, handle_distance, 180.0 - alpha)
    
    def truss2_length(self, alpha: float, v_down: float, h_abs: float) -> float:
        """Calculate truss 2 length (vertical to handle, ahead of vertical pivot).
        
        Args:
            alpha: Angle between vertical and handle (degrees)
            v_down: Vertical hole distance down from pivot (cm)
            h_abs: Handle hole absolute position from back (cm)
            
        Returns:
            Truss length (cm)
        """
        handle_distance = abs(h_abs - self.constraints.vertical_pivot_length)
        return law_of_cosines_length(v_down, handle_distance, alpha)
    
    def truss3_length(self, beta: float, h_abs: float, f_up: float) -> float:
        """Calculate truss 3 length (forearm to handle).
        
        Args:
            beta: Angle between handle and forearm (degrees)
            h_abs: Handle hole absolute position from back (cm)
            f_up: Forearm hole distance up from pivot (cm)
            
        Returns:
            Truss length (cm)
        """
        handle_distance = abs(self.constraints.forearm_pivot_length - h_abs)
        return law_of_cosines_length(handle_distance, f_up, 180.0 - beta)
    
    def solve_alpha_from_truss1(self, v_down: float, h_abs: float, truss1: float) -> float:
        """Solve alpha angle from truss 1 configuration.
        
        Args:
            v_down: Vertical hole distance down from pivot (cm)
            h_abs: Handle hole absolute position from back (cm)
            truss1: Desired truss 1 length (cm)
            
        Returns:
            Alpha angle (degrees)
        """
        handle_distance = abs(self.constraints.vertical_pivot_length - h_abs)
        theta = solve_angle_from_lengths(v_down, handle_distance, truss1)
        return 180.0 - theta
    
    def solve_alpha_from_truss2(self, v_down: float, h_abs: float, truss2: float) -> float:
        """Solve alpha angle from truss 2 configuration.
        
        Args:
            v_down: Vertical hole distance down from pivot (cm)
            h_abs: Handle hole absolute position from back (cm)
            truss2: Desired truss 2 length (cm)
            
        Returns:
            Alpha angle (degrees)
        """
        handle_distance = abs(h_abs - self.constraints.vertical_pivot_length)
        return solve_angle_from_lengths(v_down, handle_distance, truss2)
    
    def solve_beta_from_truss3(self, h_abs: float, f_up: float, truss3: float) -> float:
        """Solve beta angle from truss 3 configuration.
        
        Args:
            h_abs: Handle hole absolute position from back (cm)
            f_up: Forearm hole distance up from pivot (cm)
            truss3: Desired truss 3 length (cm)
            
        Returns:
            Beta angle (degrees)
        """
        handle_distance = abs(self.constraints.forearm_pivot_length - h_abs)
        epsilon = solve_angle_from_lengths(handle_distance, f_up, truss3)
        return 180.0 - epsilon
    
    def is_valid_geometry(self, alpha: float, beta: float) -> bool:
        """Check if angles satisfy constraints.
        
        Args:
            alpha: Angle between vertical and handle (degrees)
            beta: Angle between handle and forearm (degrees)
            
        Returns:
            True if geometry is valid
        """
        # Check individual angle bounds
        if not (self.constraints.alpha_min <= alpha <= self.constraints.alpha_max):
            return False
        if not (self.constraints.beta_min <= beta <= self.constraints.beta_max):
            return False
        
        # Check usability constraint
        if self.constraints.require_alpha_beta_sum_ge_180 and (alpha + beta < 180.0):
            return False
        
        return True
    
    def enumerate_geometries(
        self, 
        hole_layout: HoleLayout,
        target_trusses: Optional[Tuple[float, float, float]] = None,
        length_tolerance: float = 0.25
    ) -> List[Geometry]:
        """Enumerate all valid geometries for given hole layout.
        
        Args:
            hole_layout: Hole positions on rods
            target_trusses: Optional target truss lengths (T1, T2, T3)
            length_tolerance: Tolerance for truss length matching
            
        Returns:
            List of valid geometries
        """
        geometries = []
        gamma = self.constraints.forearm_pivot_length - self.constraints.vertical_pivot_length
        
        # Filter holes by position relative to pivots
        handle_left_of_v = hole_layout.handle[hole_layout.handle < self.constraints.vertical_pivot_length]
        handle_right_of_v = hole_layout.handle[hole_layout.handle > self.constraints.vertical_pivot_length]
        handle_behind_f = hole_layout.handle[hole_layout.handle < self.constraints.forearm_pivot_length]
        
        # Enumerate all combinations
        for v1 in hole_layout.vertical:
            for h1 in handle_left_of_v:
                # Calculate or check truss 1
                if target_trusses is None:
                    # Generate all possible alphas
                    for alpha in jnp.linspace(self.constraints.alpha_min, self.constraints.alpha_max, 50):
                        truss1 = self.truss1_length(alpha, v1, h1)
                        
                        # Find compatible truss 2
                        for v2 in hole_layout.vertical:
                            if v2 == v1:
                                continue
                            for h2 in handle_right_of_v:
                                truss2 = self.truss2_length(alpha, v2, h2)
                                
                                # Find compatible truss 3
                                for h3 in handle_behind_f:
                                    for f3 in hole_layout.forearm:
                                        beta = self.solve_beta_from_truss3(h3, f3, target_trusses[2] if target_trusses else 12.0)
                                        
                                        if self.is_valid_geometry(alpha, beta):
                                            truss3 = self.truss3_length(beta, h3, f3)
                                            score_alpha = angle_score_45deg(truss1, v1, abs(self.constraints.vertical_pivot_length - h1))
                                            score_beta = angle_score_45deg(truss3, abs(self.constraints.forearm_pivot_length - h3), f3)
                                            
                                            geometries.append(Geometry(
                                                alpha=round(float(alpha), 2),
                                                beta=round(float(beta), 2),
                                                gamma=round(float(gamma), 2),
                                                t1_vertical=float(v1),
                                                t1_handle=float(h1),
                                                t2_vertical=float(v2),
                                                t2_handle=float(h2),
                                                t3_handle=float(h3),
                                                t3_forearm=float(f3),
                                                truss_1=round(float(truss1), 2),
                                                truss_2=round(float(truss2), 2),
                                                truss_3=round(float(truss3), 2),
                                                score_alpha=round(float(score_alpha), 2),
                                                score_beta=round(float(score_beta), 2),
                                            ))
                else:
                    # Fixed truss lengths - solve for angles
                    alpha = self.solve_alpha_from_truss1(v1, h1, target_trusses[0])
                    if not self.is_valid_geometry(alpha, 0):  # Check alpha only
                        continue
                    
                    # Check truss 2 compatibility
                    for v2 in hole_layout.vertical:
                        if v2 == v1:
                            continue
                        for h2 in handle_right_of_v:
                            truss2_actual = self.truss2_length(alpha, v2, h2)
                            if abs(truss2_actual - target_trusses[1]) > length_tolerance:
                                continue
                            
                            # Check truss 3 compatibility
                            for h3 in handle_behind_f:
                                for f3 in hole_layout.forearm:
                                    beta = self.solve_beta_from_truss3(h3, f3, target_trusses[2])
                                    
                                    if self.is_valid_geometry(alpha, beta):
                                        score_alpha = angle_score_45deg(target_trusses[0], v1, abs(self.constraints.vertical_pivot_length - h1))
                                        score_beta = angle_score_45deg(target_trusses[2], abs(self.constraints.forearm_pivot_length - h3), f3)
                                        
                                        geometries.append(Geometry(
                                            alpha=round(float(alpha), 2),
                                            beta=round(float(beta), 2),
                                            gamma=round(float(gamma), 2),
                                            t1_vertical=float(v1),
                                            t1_handle=float(h1),
                                            t2_vertical=float(v2),
                                            t2_handle=float(h2),
                                            t3_handle=float(h3),
                                            t3_forearm=float(f3),
                                            truss_1=float(target_trusses[0]),
                                            truss_2=float(target_trusses[1]),
                                            truss_3=float(target_trusses[2]),
                                            score_alpha=round(float(score_alpha), 2),
                                            score_beta=round(float(score_beta), 2),
                                        ))
        
        return geometries


def create_uniform_holes(constraints: CrutchConstraints) -> HoleLayout:
    """Create uniformly spaced holes based on constraints.
    
    Args:
        constraints: Physical constraints
        
    Returns:
        HoleLayout with uniform spacing
    """
    def pack_1d(total_length: float, min_distance: float, margin: float) -> jnp.ndarray:
        """Pack holes uniformly in 1D."""
        positions = []
        x = margin
        while x <= total_length - margin + 1e-9:
            positions.append(x)
            x += min_distance
        return jnp.array(positions)
    
    handle_holes = pack_1d(constraints.handle_length, constraints.min_hole_distance, constraints.hole_margin)
    vertical_holes = pack_1d(constraints.vertical_length, constraints.min_hole_distance, constraints.hole_margin)
    forearm_holes = pack_1d(constraints.forearm_length, constraints.min_hole_distance, constraints.hole_margin)
    
    return HoleLayout(
        handle=handle_holes,
        vertical=vertical_holes,
        forearm=forearm_holes
    )
