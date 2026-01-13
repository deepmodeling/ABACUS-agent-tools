"""
Parameter schemas and type definitions for SCF calculations.

This module defines the schema-first approach for SCF parameters:
- Explicit type hints using Literal and Enum types
- Predefined value lists (ValueList) for all parameters
- Comprehensive documentation for each parameter
- Audit trail data structures for provenance tracking
"""

from typing import Literal, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import datetime


# ============================================================================
# ENUMERATIONS FOR ALLOWED VALUES (ValueList)
# ============================================================================

class SmearingMethod(str, Enum):
    """
    Electronic occupation smearing methods.

    Smearing is used to handle partial occupations near the Fermi level,
    which is essential for metallic systems and improves SCF convergence.
    """
    GAUSSIAN = "gaussian"
    """Gaussian smearing - safe default for most systems"""

    FERMI_DIRAC = "fd"
    """Fermi-Dirac distribution - physical at finite temperature"""

    FIXED = "fixed"
    """Fixed occupations - for molecules and insulators with gap"""

    METHFESSEL_PAXTON = "mp"
    """Methfessel-Paxton - recommended for metals, reduces energy errors"""

    MARZARI_VANDERBILT = "mv"
    """Marzari-Vanderbilt cold smearing - good for metals"""

    COLD = "cold"
    """Cold smearing - alternative for metals"""


class MixingType(str, Enum):
    """
    Charge density mixing methods for SCF convergence.

    Mixing combines old and new charge densities to achieve self-consistency.
    Different methods have different convergence properties.
    """
    PLAIN = "plain"
    """Simple linear mixing - stable but slow"""

    KERKER = "kerker"
    """Kerker preconditioning - good for metals with screening"""

    PULAY = "pulay"
    """Pulay/DIIS mixing - general purpose, good convergence"""

    PULAY_KERKER = "pulay-kerker"
    """Pulay with Kerker preconditioning - best for metals"""

    BROYDEN = "broyden"
    """Broyden mixing - alternative to Pulay"""


class BasisType(str, Enum):
    """Basis set types for electronic structure calculations."""
    PW = "pw"
    """Plane wave basis"""

    LCAO = "lcao"
    """Linear combination of atomic orbitals"""

    LCAO_IN_PW = "lcao_in_pw"
    """LCAO basis in plane wave framework"""


# ============================================================================
# PARAMETER SCHEMA WITH TYPE HINTS AND DOCUMENTATION
# ============================================================================

@dataclass
class SCFParameters:
    """
    Schema for core SCF calculation parameters.

    This dataclass defines ~15-20 core SCF parameters with:
    - Explicit type hints (Literal, Enum, float, int, bool)
    - Allowed value ranges documented in docstrings
    - Default values (None = will be filled by defaults manager)
    - Comprehensive documentation for LLM guidance

    Design principle: LLM fills this schema, doesn't generate arbitrary parameters.
    """

    # ========== Convergence Parameters ==========

    ecutwfc: Optional[float] = None
    """
    Energy cutoff for wavefunctions in Rydberg (Ry).

    - Type: float
    - Allowed values: > 0
    - Typical range: 50-150 Ry (depends on pseudopotential)
    - Default: Inferred from pseudopotential recommendations
    - Units: Rydberg (Ry), 1 Ry ≈ 13.6 eV

    Description:
        Plane-wave energy cutoff determines basis set size. Higher values
        increase accuracy but computational cost scales as O(ecutwfc^1.5).

        Guidelines:
        - Soft pseudopotentials: 50-80 Ry
        - Hard pseudopotentials: 80-120 Ry
        - Very accurate calculations: 100-150 Ry

        Always test convergence with respect to ecutwfc for production runs.
    """

    scf_thr: Optional[float] = None
    """
    SCF convergence threshold for charge density.

    - Type: float
    - Allowed values: > 0
    - Typical range: 1e-6 to 1e-9
    - Default: 1e-6
    - Units: e/Bohr³ (electron density difference)

    Description:
        Convergence criterion for self-consistent field iterations.
        SCF stops when charge density change < scf_thr.

        Guidelines:
        - Standard calculations: 1e-6
        - High accuracy (forces, phonons): 1e-7 to 1e-8
        - Very tight convergence: 1e-9 (may be slow)
        - Loose convergence: 1e-5 (for testing only)
    """

    scf_nmax: Optional[int] = None
    """
    Maximum number of SCF iterations.

    - Type: int
    - Allowed values: > 0
    - Typical range: 50-200
    - Default: 100

    Description:
        Maximum SCF steps before declaring non-convergence.
        If SCF doesn't converge within scf_nmax steps, calculation stops.

        Guidelines:
        - Standard systems: 100
        - Difficult convergence: 200-500
        - Quick tests: 50
    """

    # ========== Smearing Parameters ==========

    smearing_method: Optional[SmearingMethod] = None
    """
    Electronic occupation smearing method.

    - Type: SmearingMethod enum
    - Allowed values: gaussian, fd, fixed, mp, mv, cold
    - Default: gaussian

    Description:
        Method for smearing electronic occupations near Fermi level.
        Critical for metallic systems and SCF convergence.

        Recommendations:
        - Metals: mp (Methfessel-Paxton) or mv (Marzari-Vanderbilt)
        - Semiconductors/insulators: gaussian or fixed
        - Finite temperature: fd (Fermi-Dirac)
        - General purpose: gaussian (safe default)
    """

    smearing_sigma: Optional[float] = None
    """
    Smearing width parameter.

    - Type: float
    - Allowed values: > 0
    - Typical range: 0.001 to 0.1 Ry (0.01-1.4 eV)
    - Default: 0.015 Ry (≈ 0.2 eV)
    - Units: Rydberg (Ry)

    Description:
        Width of smearing function. Affects occupation near Fermi level.

        Guidelines:
        - Insulators with large gap: 0.001-0.01 Ry (small)
        - Semiconductors: 0.01-0.02 Ry (moderate)
        - Metals: 0.02-0.05 Ry (larger for better convergence)
        - Too large: over-smears, wrong energies
        - Too small: poor convergence

        Rule of thumb: smearing_sigma should be smaller than band gap.
    """

    # ========== Mixing Parameters ==========

    mixing_type: Optional[MixingType] = None
    """
    Charge density mixing method.

    - Type: MixingType enum
    - Allowed values: plain, kerker, pulay, pulay-kerker, broyden
    - Default: pulay

    Description:
        Method for mixing old and new charge densities in SCF iterations.

        Recommendations:
        - General purpose: pulay (good convergence)
        - Metals: pulay-kerker or kerker (handles screening)
        - Difficult systems: broyden (alternative to pulay)
        - Simple/stable: plain (slow but robust)
    """

    mixing_beta: Optional[float] = None
    """
    Mixing parameter for charge density.

    - Type: float
    - Allowed values: 0 < mixing_beta ≤ 1
    - Typical range: 0.1 to 0.8
    - Default: 0.7 (plain), 0.4 (pulay/broyden)
    - Units: dimensionless

    Description:
        Fraction of new density mixed in: ρ_new = (1-β)*ρ_old + β*ρ_new

        Guidelines:
        - Lower values (0.1-0.3): more stable, slower convergence
        - Higher values (0.5-0.8): faster but may oscillate
        - Plain mixing: 0.5-0.7
        - Pulay/Broyden: 0.3-0.5
        - Difficult convergence: reduce mixing_beta
    """

    mixing_ndim: Optional[int] = None
    """
    Mixing dimension (history size for Pulay/Broyden).

    - Type: int
    - Allowed values: > 0
    - Typical range: 4-20
    - Default: 8
    - Units: number of previous iterations

    Description:
        Number of previous iterations to use in Pulay/Broyden mixing.
        Only applies to pulay, pulay-kerker, and broyden mixing types.

        Guidelines:
        - Standard: 8
        - Memory constrained: 4-6
        - Better convergence: 10-20
        - Ignored for plain/kerker mixing
    """

    mixing_gg0: Optional[float] = None
    """
    Kerker screening parameter.

    - Type: float
    - Allowed values: ≥ 0
    - Typical range: 0.0 to 2.0
    - Default: 0.0 (no screening)
    - Units: (Bohr)⁻²

    Description:
        Screening parameter for Kerker preconditioning.
        Only applies to kerker and pulay-kerker mixing types.

        Guidelines:
        - Insulators: 0.0 (no screening needed)
        - Metals: 1.0-1.5 (improves convergence)
        - Highly metallic: 1.5-2.0
        - Ignored for plain/pulay/broyden mixing
    """

    # ========== K-point Parameters ==========

    kspacing: Optional[float] = None
    """
    K-point spacing for automatic k-mesh generation.

    - Type: float
    - Allowed values: > 0
    - Typical range: 0.1 to 0.5
    - Default: None (use KPT file instead)
    - Units: 2π/Bohr (reciprocal space)

    Description:
        Automatic k-mesh generation based on spacing.
        Alternative to providing explicit KPT file.

        Guidelines:
        - Dense mesh (accurate): 0.1-0.2
        - Standard mesh: 0.2-0.3
        - Coarse mesh (testing): 0.4-0.5
        - Smaller value = denser mesh = more k-points

        Note: Mutually exclusive with gamma_only=True
    """

    gamma_only: Optional[bool] = None
    """
    Use only Gamma point for k-sampling.

    - Type: bool
    - Allowed values: True, False
    - Default: False

    Description:
        Only use Gamma point (k=0) for Brillouin zone sampling.
        Appropriate for large supercells or isolated molecules.

        Guidelines:
        - Large supercells (>100 atoms): True
        - Molecules in box: True
        - Periodic systems: False (need k-mesh)

        Note: Mutually exclusive with kspacing
    """

    # ========== Symmetry Parameters ==========

    symmetry: Optional[bool] = None
    """
    Use crystal symmetry to reduce k-points.

    - Type: bool
    - Allowed values: True, False
    - Default: True

    Description:
        Exploit crystal symmetry to reduce computational cost.
        Symmetry reduces number of k-points in irreducible Brillouin zone.

        Guidelines:
        - Standard calculations: True (faster)
        - Symmetry-broken systems: False
        - Debugging: False (to check full k-mesh)
    """

    # ========== Output Parameters ==========

    out_chg: Optional[int] = None
    """
    Output charge density.

    - Type: int
    - Allowed values: 0 (no), 1 (yes), -1 (auto)
    - Default: 0

    Description:
        Whether to output charge density files (SPIN*_CHG).

        Options:
        - 0: Don't output charge density
        - 1: Output charge density
        - -1: Auto (output if needed for next calculation)
    """

    out_mul: Optional[bool] = None
    """
    Output Mulliken population analysis.

    - Type: bool
    - Allowed values: True, False
    - Default: False

    Description:
        Perform Mulliken population analysis and output results.
        Provides atomic charges and orbital populations.

        Note: Only available for LCAO basis
    """

    # ========== Advanced Parameters ==========

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
# AUDIT TRAIL DATA STRUCTURES
# ============================================================================

@dataclass
class ParameterProvenance:
    """
    Tracks the origin and reasoning for each parameter value.

    This provides full traceability: every parameter has documented provenance
    showing where it came from and why it has its current value.
    """

    parameter_name: str
    """Name of the parameter (e.g., 'ecutwfc', 'mixing_beta')"""

    value: Any
    """Current value of the parameter"""

    source: Literal["user_input", "default", "inferred", "dependency"]
    """
    Source of the parameter value:
    - user_input: Explicitly provided by user
    - default: Standard default value
    - inferred: Inferred from other parameters via rules
    - dependency: Set due to dependency constraint
    """

    reasoning: str
    """Human-readable explanation of why this value was chosen"""

    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    """ISO timestamp when this provenance was recorded"""

    # For dependency-based and inferred values
    depends_on: Optional[List[str]] = None
    """List of parameter names this value depends on (if source=inferred/dependency)"""

    inference_rule: Optional[str] = None
    """Name of the inference rule applied (if source=inferred)"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "parameter_name": self.parameter_name,
            "value": self.value,
            "source": self.source,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "depends_on": self.depends_on,
            "inference_rule": self.inference_rule,
        }


@dataclass
class SCFAuditTrail:
    """
    Complete audit trail for an SCF calculation.

    Contains all parameter provenances, validation results, warnings, and errors.
    Provides full traceability from user intent to final ABACUS INPUT parameters.
    """

    calculation_id: str
    """Unique identifier for this calculation"""

    parameters: dict[str, ParameterProvenance]
    """Dictionary mapping parameter names to their provenance"""

    validation_results: List[dict]
    """List of validation results (errors, warnings, info)"""

    warnings: List[str]
    """List of warning messages"""

    errors: List[str]
    """List of error messages"""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "calculation_id": self.calculation_id,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "validation_results": self.validation_results,
            "warnings": self.warnings,
            "errors": self.errors,
        }
