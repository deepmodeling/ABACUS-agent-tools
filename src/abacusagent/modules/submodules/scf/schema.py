"""
Parameter schemas and type definitions for SCF calculations.

This module defines the schema-first approach for SCF parameters:
- Explicit type hints using Literal and Enum types
- Predefined value lists (ValueList) for all parameters
- Comprehensive documentation for each parameter
- Inherits common parameters from the shared framework
"""

from typing import Literal, Optional
from dataclasses import dataclass

# Import common enums and parameter groups from shared framework
from ..common import (
    SmearingMethod,
    MixingType,
    BasisType,
    CommonSCFParameters,
)


# ============================================================================
# SCF-SPECIFIC PARAMETER SCHEMA
# ============================================================================

@dataclass
class SCFParameters(CommonSCFParameters):
    """
    Schema for SCF calculation parameters.

    Inherits common SCF parameters from CommonSCFParameters:
    - Convergence: ecutwfc, scf_thr, scf_nmax
    - Smearing: smearing_method, smearing_sigma
    - Mixing: mixing_type, mixing_beta, mixing_ndim, mixing_gg0
    - K-points: kspacing, gamma_only
    - Output: symmetry, out_chg, out_mul

    Adds SCF-specific parameters:
    - chg_extrap: Charge density extrapolation method
    - ks_solver: Kohn-Sham equation solver
    - nspin: Number of spin channels

    Design principle: LLM fills this schema, doesn't generate arbitrary parameters.
    """

    # ========== Advanced Parameters (SCF-specific) ==========

    chg_extrap: Optional[Literal["none", "atomic", "first-order", "second-order"]] = None
    """
    Charge density extrapolation method.

    - Type: Literal
    - Allowed values: none, atomic, first-order, second-order
    - Default: atomic

    Description:
        Method for extrapolating charge density in relaxation/MD.

        Options:
        - none: No extrapolation (start from atomic)
        - atomic: Use atomic charge density
        - first-order: Linear extrapolation from previous step
        - second-order: Quadratic extrapolation from previous steps

        Note: Only relevant for relaxation/MD, not single-point SCF
    """

    ks_solver: Optional[Literal["cg", "dav", "bpcg", "genelpa", "scalapack_gvx"]] = None
    """
    Kohn-Sham equation solver.

    - Type: Literal
    - Allowed values: cg, dav, bpcg, genelpa, scalapack_gvx
    - Default: cg (PW basis), genelpa (LCAO basis)

    Description:
        Eigenvalue solver algorithm for Kohn-Sham equations.

        Options:
        - cg: Conjugate gradient (default for PW)
        - dav: Davidson diagonalization
        - bpcg: Block preconditioned conjugate gradient
        - genelpa: ELPA library (default for LCAO, parallel)
        - scalapack_gvx: ScaLAPACK solver (parallel)
    """

    # ========== Spin Parameters (for reference) ==========

    nspin: Optional[Literal[1, 2, 4]] = None
    """
    Number of spin channels.

    - Type: Literal[1, 2, 4]
    - Allowed values: 1 (non-spin), 2 (collinear), 4 (non-collinear)
    - Default: 1

    Description:
        Spin polarization setting.

        Options:
        - 1: Non-spin-polarized (closed shell, non-magnetic)
        - 2: Spin-polarized collinear (magnetic, spin up/down)
        - 4: Non-collinear spin (spin-orbit coupling, complex magnetism)

    Note: Usually set via abacus_prepare(), included here for completeness.
          If soc=True, nspin must be 4.
    """


# ============================================================================
# RE-EXPORT COMMON TYPES FOR BACKWARD COMPATIBILITY
# ============================================================================

# Re-export enums so existing code can still import from this module
__all__ = [
    "SCFParameters",
    "SmearingMethod",
    "MixingType",
    "BasisType",
]
