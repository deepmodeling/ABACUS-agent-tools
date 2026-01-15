"""
Shared parameter definitions used across multiple calculation modules.

This module defines common enums and parameter groups that are reused across
different calculation types (SCF, relax, band, DOS, MD, etc.).
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


# ============================================================================
# Common Enums (ValueLists)
# ============================================================================

class SmearingMethod(str, Enum):
    """
    Electronic occupation smearing methods.

    Used to handle partial occupancies near the Fermi level, especially
    important for metallic systems.

    Values:
        GAUSSIAN: Gaussian smearing (safe default for most systems)
        FERMI_DIRAC: Fermi-Dirac distribution (physical at finite temperature)
        FIXED: Fixed occupancies (for insulators with large gap)
        METHFESSEL_PAXTON: Methfessel-Paxton method (recommended for metals)
        MARZARI_VANDERBILT: Marzari-Vanderbilt cold smearing (for metals)
        COLD: Cold smearing (alternative for metals)
    """
    GAUSSIAN = "gaussian"
    FERMI_DIRAC = "fd"
    FIXED = "fixed"
    METHFESSEL_PAXTON = "mp"
    MARZARI_VANDERBILT = "mv"
    COLD = "cold"


class MixingType(str, Enum):
    """
    Charge density mixing methods for SCF convergence.

    Different mixing schemes have different convergence properties and
    are suited for different types of systems.

    Values:
        PLAIN: Simple linear mixing (basic, may be slow)
        KERKER: Kerker mixing (good for metals with screening)
        PULAY: Pulay mixing (general purpose, good convergence)
        PULAY_KERKER: Pulay with Kerker screening (best for metals)
        BROYDEN: Broyden mixing (alternative to Pulay)
    """
    PLAIN = "plain"
    KERKER = "kerker"
    PULAY = "pulay"
    PULAY_KERKER = "pulay-kerker"
    BROYDEN = "broyden"


class BasisType(str, Enum):
    """
    Basis set types for electronic structure calculations.

    Values:
        PW: Plane wave basis (systematic, good for periodic systems)
        LCAO: Linear combination of atomic orbitals (efficient for large systems)
        LCAO_IN_PW: LCAO basis in plane wave framework (hybrid approach)
    """
    PW = "pw"
    LCAO = "lcao"
    LCAO_IN_PW = "lcao_in_pw"


# ============================================================================
# Common Parameter Groups
# ============================================================================

@dataclass
class ConvergenceParameters:
    """
    Convergence parameters for SCF calculations.

    These parameters control the convergence criteria and iteration limits
    for self-consistent field calculations.

    Attributes:
        scf_thr: SCF convergence threshold (energy difference in eV)
            Typical: 1e-6 for standard calculations, 1e-8 for tight convergence
        scf_nmax: Maximum number of SCF iterations
            Typical: 100 for standard calculations, 200-300 for difficult systems
        ecutwfc: Plane wave energy cutoff in Rydberg (Ry)
            Typical: 50-150 Ry depending on pseudopotentials
            Note: 1 Ry ≈ 13.6 eV
    """
    scf_thr: Optional[float] = None
    scf_nmax: Optional[int] = None
    ecutwfc: Optional[float] = None


@dataclass
class SmearingParameters:
    """
    Electronic occupation smearing parameters.

    Smearing is used to handle partial occupancies near the Fermi level,
    which is essential for metallic systems and improves convergence.

    Attributes:
        smearing_method: Method for electronic occupation smearing
            See SmearingMethod enum for available options
        smearing_sigma: Smearing width in Rydberg (Ry)
            Typical: 0.01-0.02 Ry (≈0.14-0.27 eV) for metals
                     0.001-0.005 Ry for semiconductors
            Note: 1 Ry ≈ 13.6 eV
    """
    smearing_method: Optional[SmearingMethod] = None
    smearing_sigma: Optional[float] = None


@dataclass
class MixingParameters:
    """
    Charge density mixing parameters for SCF convergence.

    Mixing controls how the new charge density is combined with the old
    density in each SCF iteration. Proper mixing is crucial for convergence.

    Attributes:
        mixing_type: Mixing method for charge density
            See MixingType enum for available options
        mixing_beta: Mixing parameter (0 < beta ≤ 1)
            Typical: 0.7 for plain/kerker, 0.4 for pulay/broyden
            Lower values = more stable but slower convergence
        mixing_ndim: Dimension of mixing history (for pulay/broyden)
            Typical: 8 for standard calculations
            Only used with pulay, pulay-kerker, or broyden mixing
        mixing_gg0: Kerker screening parameter (for kerker-based mixing)
            Typical: 0.0-2.0, higher for more metallic systems
            Only used with kerker or pulay-kerker mixing
    """
    mixing_type: Optional[MixingType] = None
    mixing_beta: Optional[float] = None
    mixing_ndim: Optional[int] = None
    mixing_gg0: Optional[float] = None


@dataclass
class KPointParameters:
    """
    K-point sampling parameters.

    K-points are used to sample the Brillouin zone in periodic systems.
    Proper k-point sampling is essential for accurate results.

    Attributes:
        kspacing: Automatic k-point spacing in 1/Å
            Typical: 0.1-0.5 Å⁻¹
            Smaller values = denser mesh = more accurate but slower
            Mutually exclusive with gamma_only
        gamma_only: Use only the Gamma point (k=0)
            Suitable for large supercells or molecules
            Mutually exclusive with kspacing
    """
    kspacing: Optional[float] = None
    gamma_only: Optional[bool] = None


@dataclass
class ForceStressParameters:
    """
    Force and stress convergence thresholds.

    Used in geometry optimization and cell relaxation calculations.

    Attributes:
        force_thr_ev: Force convergence threshold in eV/Å
            Typical: 0.01-0.05 eV/Å for standard relaxation
                     0.001-0.005 eV/Å for tight relaxation
        stress_thr: Stress convergence threshold in kBar
            Typical: 0.1-1.0 kBar for cell relaxation
            Only relevant when relaxing cell parameters
    """
    force_thr_ev: Optional[float] = None
    stress_thr: Optional[float] = None


@dataclass
class OutputParameters:
    """
    Output control parameters.

    These parameters control what data is written to output files.

    Attributes:
        symmetry: Use crystal symmetry to reduce k-points and speed up calculation
            Default: True (recommended for most cases)
        out_chg: Output charge density
            -1: output charge density at every SCF step
             0: do not output charge density (default)
             1: output final charge density
        out_mul: Output Mulliken population analysis
            Only works with LCAO basis
            Default: False
    """
    symmetry: Optional[bool] = None
    out_chg: Optional[int] = None
    out_mul: Optional[bool] = None
