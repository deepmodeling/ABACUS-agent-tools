"""
Composable parameter groups for building module-specific schemas.

This module provides pre-composed parameter groups that combine multiple
basic parameter groups. Module-specific schemas can inherit from these
to quickly build their parameter sets.
"""

from dataclasses import dataclass
from typing import Optional
from .shared_parameters import (
    ConvergenceParameters,
    SmearingParameters,
    MixingParameters,
    KPointParameters,
    ForceStressParameters,
    OutputParameters,
)


@dataclass
class CommonSCFParameters(
    ConvergenceParameters,
    SmearingParameters,
    MixingParameters,
    KPointParameters,
    OutputParameters
):
    """
    Common SCF-related parameters used by multiple calculation types.

    This combines convergence, smearing, mixing, k-point, and output parameters
    that are shared across SCF, relax, band, DOS, MD, and elastic calculations.

    Inherits from:
        - ConvergenceParameters: scf_thr, scf_nmax, ecutwfc
        - SmearingParameters: smearing_method, smearing_sigma
        - MixingParameters: mixing_type, mixing_beta, mixing_ndim, mixing_gg0
        - KPointParameters: kspacing, gamma_only
        - OutputParameters: symmetry, out_chg, out_mul

    Usage:
        Module-specific parameter classes can inherit from this to get all
        common SCF parameters, then add their own module-specific parameters.

        Example:
            @dataclass
            class RelaxParameters(CommonSCFParameters, ForceStressParameters):
                # Relax-specific parameters
                relax_nmax: Optional[int] = None
                relax_method: Optional[str] = None
    """
    pass


@dataclass
class CommonRelaxationParameters(
    ConvergenceParameters,
    SmearingParameters,
    MixingParameters,
    KPointParameters,
    ForceStressParameters,
    OutputParameters
):
    """
    Common parameters for relaxation-type calculations.

    This extends CommonSCFParameters with force/stress thresholds,
    suitable for geometry optimization and cell relaxation.

    Inherits from:
        - ConvergenceParameters: scf_thr, scf_nmax, ecutwfc
        - SmearingParameters: smearing_method, smearing_sigma
        - MixingParameters: mixing_type, mixing_beta, mixing_ndim, mixing_gg0
        - KPointParameters: kspacing, gamma_only
        - ForceStressParameters: force_thr_ev, stress_thr
        - OutputParameters: symmetry, out_chg, out_mul

    Usage:
        Suitable for relax, elastic, and EOS calculations that need both
        SCF convergence and force/stress thresholds.

        Example:
            @dataclass
            class ElasticParameters(CommonRelaxationParameters):
                # Elastic-specific parameters
                norm_strain: Optional[float] = None
                shear_strain: Optional[float] = None
    """
    pass


@dataclass
class CommonPostSCFParameters(
    ConvergenceParameters,
    SmearingParameters,
    MixingParameters,
    KPointParameters,
    OutputParameters
):
    """
    Common parameters for post-SCF calculations.

    This is identical to CommonSCFParameters but semantically indicates
    calculations that run after an initial SCF (e.g., band, DOS).

    Inherits from:
        - ConvergenceParameters: scf_thr, scf_nmax, ecutwfc
        - SmearingParameters: smearing_method, smearing_sigma
        - MixingParameters: mixing_type, mixing_beta, mixing_ndim, mixing_gg0
        - KPointParameters: kspacing, gamma_only
        - OutputParameters: symmetry, out_chg, out_mul

    Usage:
        Suitable for band and DOS calculations that may need to run
        an initial SCF before the main calculation.

        Example:
            @dataclass
            class BandParameters(CommonPostSCFParameters):
                # Band-specific parameters
                kpath: Optional[List[str]] = None
                energy_min: Optional[float] = None
    """
    pass
