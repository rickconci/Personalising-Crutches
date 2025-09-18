# crutch_geometry.py
# ---------------------------------------------------------
# Pydantic + JAX implementation with your coordinate conventions
#
# Coordinate conventions:
# - vertical holes: distance DOWN from the vertical pivot (cm)
# - forearm holes: distance UP from the forearm pivot (cm)
# - handle holes: absolute position from the BACK of the handle rod, forward (cm)
# - vertical_pivot_length and forearm_pivot_length: absolute positions on the handle axis from the BACK
#
# Truss mapping:
#   truss_1: VERTICAL ↔ HANDLE hole with handle_pos < vertical_pivot_length
#            included angle = (180 - α)
#   truss_2: VERTICAL ↔ HANDLE hole with handle_pos > vertical_pivot_length
#            included angle = α
#   truss_3: FOREARM  ↔ HANDLE hole with handle_pos < forearm_pivot_length
#            included angle = (180 - β)
#
# Angles (forward side):
#   α = angle between vertical and handle
#   β = angle between handle and forearm
# Usability constraint: α + β ≥ 180°

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
import jax.numpy as jnp

# -------------------------------
# Helpers
# -------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(jnp.clip(x, lo, hi))

def law_of_cosines_length(r1: float, r2: float, theta_deg: float) -> float:
    """Distance between endpoints of two spokes r1, r2 with included angle theta (deg)."""
    th = jnp.deg2rad(theta_deg)
    return float(jnp.sqrt(jnp.maximum(0.0, r1*r1 + r2*r2 - 2.0*r1*r2*jnp.cos(th))))

def solve_angle_from_lengths(a: float, b: float, c: float) -> float:
    """Solve the included angle (deg) between a and b given opposite side c."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    num = a*a + b*b - c*c
    den = 2.0*a*b
    x = _clamp(num/den, -1.0, 1.0)
    return float(jnp.degrees(jnp.arccos(x)))

def angle_score_45deg(L: float, r1: float, r2: float) -> float:
    """Preference for near-45° angle between truss and each spoke."""
    def angle_at_side(a, b, c):
        if b == 0.0 or c == 0.0:
            return 90.0
        x = _clamp((b*b + c*c - a*a) / (2.0*b*c), -1.0, 1.0)
        return float(jnp.degrees(jnp.arccos(x)))
    phi1 = angle_at_side(r2, L, r1)
    phi2 = angle_at_side(r1, L, r2)
    return abs(phi1 - 45.0) + abs(phi2 - 45.0)

# -------------------------------
# Data models (pydantic)
# -------------------------------

class Constraints(BaseModel):
    vertical_length: float = Field(20.0, gt=0)
    handle_length: float = Field(38.0, gt=0)
    forearm_length: float = Field(17.0, gt=0)

    vertical_pivot_length: float = Field(19.0, ge=0)
    forearm_pivot_length: float = Field(19.0, ge=0)

    min_hole_distance: float = Field(2.0, gt=0)
    hole_margin: float = Field(0.5, ge=0)

    alpha_min: float = 85.0
    alpha_max: float = 115.0
    beta_min: float  = 95.0
    beta_max: float  = 140.0

    gamma_min: float = -9.0
    gamma_max: float = 9.0

    @validator("vertical_pivot_length", "forearm_pivot_length")
    def pivots_inside_handle(cls, v, values):
        handle_length = values.get("handle_length", 38.0)
        if not (0.0 <= v <= handle_length):
            raise ValueError("pivot must lie within [0, handle_length]")
        return v

    @validator("alpha_max")
    def check_alpha(cls, v, values):
        if v < values.get("alpha_min", 85.0):
            raise ValueError("alpha_max < alpha_min")
        return v

    @validator("beta_max")
    def check_beta(cls, v, values):
        if v < values.get("beta_min", 95.0):
            raise ValueError("beta_max < beta_min")
        return v

class TrussSet(BaseModel):
    truss_1: float = Field(..., gt=0)
    truss_2: float = Field(..., gt=0)
    truss_3: float = Field(..., gt=0)

class HoleLayout(BaseModel):
    handle: List[float]
    vertical: List[float]
    forearm: List[float]

class Geometry(BaseModel):
    alpha: float
    beta: float
    gamma: float
    t1_vertical: float
    t1_handle: float
    t2_vertical: float
    t2_handle: float
    t3_handle: float
    t3_forearm: float
    score_alpha: float
    score_beta: float

# -------------------------------
# Crutch model
# -------------------------------

class Crutch(BaseModel):
    constraints: Constraints
    hole_layout: Optional[HoleLayout] = None

    class Config:
        arbitrary_types_allowed = True

    def set_holes(self) -> HoleLayout:
        c = self.constraints
        step = c.min_hole_distance
        margin = c.hole_margin
        def pack_1d(total_len: float) -> List[float]:
            x = margin
            holes = []
            while x <= total_len - margin + 1e-9:
                holes.append(round(float(x), 3))
                x += step
            return holes
        handle = pack_1d(c.handle_length)
        vertical = pack_1d(c.vertical_length)
        forearm = pack_1d(c.forearm_length)
        self.hole_layout = HoleLayout(handle=handle, vertical=vertical, forearm=forearm)
        return self.hole_layout

    # Truss length calculators with corrected included angles
    def truss1_len(self, alpha: float, v_down: float, h_abs: float) -> float:
        return law_of_cosines_length(v_down, abs(self.constraints.vertical_pivot_length - h_abs), 180.0 - alpha)

    def truss2_len(self, alpha: float, v_down: float, h_abs: float) -> float:
        return law_of_cosines_length(v_down, abs(h_abs - self.constraints.vertical_pivot_length), alpha)

    def truss3_len(self, beta: float, h_abs: float, f_up: float) -> float:
        return law_of_cosines_length(abs(self.constraints.forearm_pivot_length - h_abs), f_up, 180.0 - beta)

    # Solvers
    def solve_alpha_from_t1(self, v_down: float, h_abs: float, T1: float) -> float:
        theta = solve_angle_from_lengths(v_down, abs(self.constraints.vertical_pivot_length - h_abs), T1)
        return 180.0 - theta

    def solve_alpha_from_t2(self, v_down: float, h_abs: float, T2: float) -> float:
        return solve_angle_from_lengths(v_down, abs(h_abs - self.constraints.vertical_pivot_length), T2)

    def solve_beta_from_t3(self, h_abs: float, f_up: float, T3: float) -> float:
        epsilon = solve_angle_from_lengths(abs(self.constraints.forearm_pivot_length - h_abs), f_up, T3)
        return 180.0 - epsilon

    # Enumeration
    def enumerate_geometries_fixed_trusses(
        self,
        trusses: TrussSet,
        length_tol: float = 0.25,
        require_alpha_beta_sum_ge_180: bool = True,
    ) -> List[Geometry]:
        if self.hole_layout is None:
            self.set_holes()
        c = self.constraints
        H, V, F = self.hole_layout.handle, self.hole_layout.vertical, self.hole_layout.forearm
        vpiv, fpiv = c.vertical_pivot_length, c.forearm_pivot_length
        gamma_val = fpiv - vpiv

        handle_left_of_v  = [h for h in H if h < vpiv]
        handle_right_of_v = [h for h in H if h > vpiv]
        handle_behind_f   = [h for h in H if h < fpiv]

        out: List[Geometry] = []

        # Step 1: seed with Truss1 (back)
        for v1 in V:
            for h1 in handle_left_of_v:
                alpha = self.solve_alpha_from_t1(v1, h1, trusses.truss_1)
                if not (c.alpha_min <= alpha <= c.alpha_max):
                    continue
                # Step 2: check Truss2 at same alpha
                for v2 in V:
                    if v2 == v1: continue
                    for h2 in handle_right_of_v:
                        L2 = self.truss2_len(alpha, v2, h2)
                        if abs(L2 - trusses.truss_2) > length_tol: continue
                        # Step 3: solve beta from Truss3
                        for h3 in handle_behind_f:
                            for f3 in F:
                                beta = self.solve_beta_from_t3(h3, f3, trusses.truss_3)
                                if not (c.beta_min <= beta <= c.beta_max): continue
                                if require_alpha_beta_sum_ge_180 and (alpha + beta < 180.0): continue
                                s_a = angle_score_45deg(trusses.truss_1, v1, abs(vpiv - h1))
                                s_b = angle_score_45deg(trusses.truss_3, abs(fpiv - h3), f3)
                                out.append(Geometry(
                                    alpha=round(alpha,2),
                                    beta=round(beta,2),
                                    gamma=round(gamma_val,2),
                                    t1_vertical=v1,
                                    t1_handle=h1,
                                    t2_vertical=v2,
                                    t2_handle=h2,
                                    t3_handle=h3,
                                    t3_forearm=f3,
                                    score_alpha=round(s_a,2),
                                    score_beta=round(s_b,2),
                                ))
        return out

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    constraints = Constraints()
    crutch = Crutch(constraints=constraints)
    crutch.set_holes()
    trusses = TrussSet(truss_1=12.0, truss_2=12.0, truss_3=12.0)
    geoms = crutch.enumerate_geometries_fixed_trusses(trusses)
    print("Found", len(geoms), "feasible geometries")
    for g in geoms[:5]:
        print(g.dict())