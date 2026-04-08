"""
Parameter schemas and type definitions for relax calculations.

This module defines the schema-first approach for relax parameters:
- Inherits common parameters from the shared framework
- Adds relax-specific parameters
- Explicit type hints using Literal types
- Comprehensive documentation for each parameter
"""

from typing import Literal, Optional
from dataclasses import dataclass

# Import common parameter groups from shared framework
from ..common import CommonRelaxationParameters


# ============================================================================
# RELAX-SPECIFIC PARAMETER SCHEMA
# ============================================================================

@dataclass
class RelaxParameters(CommonRelaxationParameters):
    """
    Schema for relax calculation parameters.

    Inherits common parameters from CommonRelaxationParameters:
    - Convergence: ecutwfc, scf_thr, scf_nmax
    - Smearing: smearing_method, smearing_sigma
    - Mixing: mixing_type, mixing_beta, mixing_ndim, mixing_gg0
    - K-points: kspacing, gamma_only
    - Force/Stress: force_thr_ev, stress_thr
    - Output: symmetry, out_chg, out_mul

    Adds relax-specific parameters:
    - relax_nmax: Maximum number of relaxation steps
    - relax_method: Relaxation algorithm
    - relax_new: Use new CG implementation
    - relax_cell: Whether to relax cell parameters
    - fixed_axes: Which axes to fix during cell relaxation

    Design principle: LLM fills this schema, doesn't generate arbitrary parameters.
    """

    # ========== Relaxation Control Parameters ==========

    relax_nmax: Optional[int] = None
    """
    Maximum number of relaxation steps.

    - Type: int
    - Allowed values: > 0
    - Typical range: 50-200
    - Default: 100

    Description:
        Maximum number of ionic relaxation steps before stopping.
        If relaxation doesn't converge within relax_nmax steps, calculation stops.

        Guidelines:
        - Standard systems: 100
        - Difficult relaxation: 200-500
        - Quick tests: 50
    """

    relax_method: Optional[Literal["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]] = None
    """
    Relaxation algorithm.

    - Type: Literal
    - Allowed values: cg, bfgs, bfgs_trad, cg_bfgs, sd, fire
    - Default: cg

    Description:
        Algorithm for ionic relaxation.

        Options:
        - cg: Conjugate gradient (default, reliable)
        - bfgs: BFGS quasi-Newton (fast for well-behaved systems)
        - bfgs_trad: Traditional BFGS implementation
        - cg_bfgs: Hybrid CG and BFGS
        - sd: Steepest descent (slow but robust)
        - fire: Fast inertial relaxation engine (good for large systems)

        Recommendations:
        - General purpose: cg
        - Fast convergence: bfgs
        - Difficult systems: sd or fire
    """

    relax_new: Optional[bool] = None
    """
    Use new CG implementation.

    - Type: bool
    - Allowed values: True, False
    - Default: True

    Description:
        Whether to use the new implemented CG method.
        Only relevant when relax_method='cg'.

        Guidelines:
        - Standard: True (recommended)
        - Compatibility: False (use old implementation)
    """

    # ========== Cell Relaxation Parameters ==========

    relax_cell: Optional[bool] = None
    """
    Whether to relax cell parameters.

    - Type: bool
    - Allowed values: True, False
    - Default: False

    Description:
        If True, performs cell-relax (optimize both atomic positions and cell).
        If False, performs relax (optimize only atomic positions).

        Guidelines:
        - Atomic positions only: False
        - Full structure optimization: True

        Note: When True, stress_thr becomes relevant
    """

    fixed_axes: Optional[Literal["None", "volume", "shape", "a", "b", "c", "ab", "ac", "bc"]] = None
    """
    Which axes to fix during cell relaxation.

    - Type: Literal
    - Allowed values: None, volume, shape, a, b, c, ab, ac, bc
    - Default: None (relax all axes)

    Description:
        Specifies constraints on cell relaxation.
        Only effective when relax_cell=True.

        Options:
        - None: Relax all axes (default)
        - volume: Fixed volume, relax shape
        - shape: Fixed shape, relax volume (only lattice constant changes)
        - a: Fix a axis
        - b: Fix b axis
        - c: Fix c axis
        - ab: Fix both a and b axes
        - ac: Fix both a and c axes
        - bc: Fix both b and c axes

        Guidelines:
        - Full relaxation: None
        - Constant volume: volume
        - Preserve symmetry: shape
        - 2D materials: c (fix out-of-plane)
    """


# ============================================================================
# RE-EXPORT COMMON TYPES FOR BACKWARD COMPATIBILITY
# ============================================================================

# Re-export enums so existing code can still import from this module
from ..common import SmearingMethod, MixingType

__all__ = [
    "RelaxParameters",
    "SmearingMethod",
    "MixingType",
]
