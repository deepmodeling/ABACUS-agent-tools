"""
Base validator providing common validation methods.

This module provides the BaseParameterValidator class that all module-specific
validators inherit from. It includes validation methods for common parameters
like convergence, smearing, mixing, and k-points.
"""

from typing import Dict, List, Tuple, Any, Optional
from .base_schema import BaseParameters, ValidationResult


class BaseParameterValidator:
    """
    Base validator providing common validation methods.

    Module-specific validators inherit from this and add their own rules.
    This ensures consistent validation of common parameters across all modules.

    Attributes:
        validation_results: List of all validation results
        warnings: List of warning messages
        errors: List of error messages
    """

    def __init__(self):
        """Initialize validator with empty result lists."""
        self.validation_results: List[ValidationResult] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def validate_all(
        self,
        params: BaseParameters,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.

        Must be implemented by subclasses to define the full validation workflow.

        Args:
            params: Parameter object to validate
            context: Context dictionary with additional information
                (e.g., basis_type, soc, existing INPUT parameters)

        Returns:
            Tuple of (is_valid, validation_results)
            is_valid is True if no errors (warnings are allowed)
        """
        raise NotImplementedError("Subclasses must implement validate_all()")

    # ========================================================================
    # Common Validation Methods
    # ========================================================================

    def _validate_convergence_params(self, params: Any):
        """
        Validate common convergence parameters.

        Checks scf_thr, scf_nmax, and ecutwfc for valid ranges.

        Args:
            params: Parameter object with convergence attributes
        """
        # Validate scf_thr
        if hasattr(params, 'scf_thr') and params.scf_thr is not None:
            if params.scf_thr <= 0:
                self._add_error('scf_thr', f"scf_thr must be > 0, got {params.scf_thr}")
            elif params.scf_thr > 1e-3:
                self._add_warning('scf_thr',
                                f"scf_thr={params.scf_thr} is loose, may result in inaccurate results")
            elif params.scf_thr < 1e-12:
                self._add_warning('scf_thr',
                                f"scf_thr={params.scf_thr} is very tight, may be difficult to converge")
            else:
                self._add_info('scf_thr', f"scf_thr={params.scf_thr} is within typical range")

        # Validate scf_nmax
        if hasattr(params, 'scf_nmax') and params.scf_nmax is not None:
            if params.scf_nmax <= 0:
                self._add_error('scf_nmax', f"scf_nmax must be > 0, got {params.scf_nmax}")
            elif params.scf_nmax < 20:
                self._add_warning('scf_nmax',
                                f"scf_nmax={params.scf_nmax} is low, may not converge")
            else:
                self._add_info('scf_nmax', f"scf_nmax={params.scf_nmax} is sufficient")

        # Validate ecutwfc
        if hasattr(params, 'ecutwfc') and params.ecutwfc is not None:
            if params.ecutwfc <= 0:
                self._add_error('ecutwfc', f"ecutwfc must be > 0, got {params.ecutwfc}")
            elif params.ecutwfc < 20:
                self._add_warning('ecutwfc',
                                f"ecutwfc={params.ecutwfc} Ry is very low, results may be inaccurate")
            elif params.ecutwfc > 200:
                self._add_warning('ecutwfc',
                                f"ecutwfc={params.ecutwfc} Ry is very high, may be unnecessarily expensive")
            else:
                self._add_info('ecutwfc', f"ecutwfc={params.ecutwfc} Ry is within typical range")

    def _validate_smearing_params(self, params: Any):
        """
        Validate common smearing parameters.

        Checks smearing_method and smearing_sigma for valid values and dependencies.

        Args:
            params: Parameter object with smearing attributes
        """
        # Validate smearing_sigma
        if hasattr(params, 'smearing_sigma') and params.smearing_sigma is not None:
            if params.smearing_sigma <= 0:
                self._add_error('smearing_sigma',
                              f"smearing_sigma must be > 0, got {params.smearing_sigma}")
            elif params.smearing_sigma > 0.1:
                self._add_warning('smearing_sigma',
                                f"smearing_sigma={params.smearing_sigma} Ry is large, "
                                "may over-smear the Fermi surface")
            elif params.smearing_sigma < 0.001:
                self._add_warning('smearing_sigma',
                                f"smearing_sigma={params.smearing_sigma} Ry is small, "
                                "may cause convergence issues")
            else:
                self._add_info('smearing_sigma',
                             f"smearing_sigma={params.smearing_sigma} Ry is within typical range")

        # Check dependency: smearing_sigma should be provided if method is not 'fixed'
        if hasattr(params, 'smearing_method') and hasattr(params, 'smearing_sigma'):
            if params.smearing_method is not None and params.smearing_method != "fixed":
                if params.smearing_sigma is None:
                    self._add_warning('smearing_sigma',
                                    f"smearing_method={params.smearing_method} typically requires "
                                    "smearing_sigma to be specified")

    def _validate_mixing_params(self, params: Any):
        """
        Validate common mixing parameters.

        Checks mixing_type, mixing_beta, mixing_ndim, and mixing_gg0 for
        valid ranges and dependencies.

        Args:
            params: Parameter object with mixing attributes
        """
        # Validate mixing_beta
        if hasattr(params, 'mixing_beta') and params.mixing_beta is not None:
            if params.mixing_beta <= 0 or params.mixing_beta > 1:
                self._add_error('mixing_beta',
                              f"mixing_beta must be in (0, 1], got {params.mixing_beta}")
            elif params.mixing_beta > 0.8:
                self._add_warning('mixing_beta',
                                f"mixing_beta={params.mixing_beta} is high, may cause instability")
            elif params.mixing_beta < 0.1:
                self._add_warning('mixing_beta',
                                f"mixing_beta={params.mixing_beta} is low, convergence may be slow")
            else:
                self._add_info('mixing_beta', f"mixing_beta={params.mixing_beta} is within typical range")

        # Validate mixing_ndim
        if hasattr(params, 'mixing_ndim') and params.mixing_ndim is not None:
            if params.mixing_ndim <= 0:
                self._add_error('mixing_ndim', f"mixing_ndim must be > 0, got {params.mixing_ndim}")
            elif params.mixing_ndim > 20:
                self._add_warning('mixing_ndim',
                                f"mixing_ndim={params.mixing_ndim} is large, may use excessive memory")
            elif params.mixing_ndim < 4:
                self._add_warning('mixing_ndim',
                                f"mixing_ndim={params.mixing_ndim} is small, may converge slowly")
            else:
                self._add_info('mixing_ndim', f"mixing_ndim={params.mixing_ndim} is within typical range")

        # Validate mixing_gg0
        if hasattr(params, 'mixing_gg0') and params.mixing_gg0 is not None:
            if params.mixing_gg0 < 0:
                self._add_error('mixing_gg0', f"mixing_gg0 must be ≥ 0, got {params.mixing_gg0}")
            else:
                self._add_info('mixing_gg0', f"mixing_gg0={params.mixing_gg0} is valid")

        # Check dependencies
        if hasattr(params, 'mixing_type') and params.mixing_type is not None:
            mixing_type_str = params.mixing_type.value if hasattr(params.mixing_type, 'value') else str(params.mixing_type)

            # mixing_ndim only applies to pulay/broyden/pulay-kerker
            if hasattr(params, 'mixing_ndim') and params.mixing_ndim is not None:
                if mixing_type_str not in ["pulay", "broyden", "pulay-kerker"]:
                    self._add_info('mixing_ndim',
                                 f"mixing_ndim is only used with pulay/broyden/pulay-kerker mixing, "
                                 f"but mixing_type={mixing_type_str}")

            # mixing_gg0 only applies to kerker/pulay-kerker
            if hasattr(params, 'mixing_gg0') and params.mixing_gg0 is not None:
                if mixing_type_str not in ["kerker", "pulay-kerker"]:
                    self._add_info('mixing_gg0',
                                 f"mixing_gg0 is only used with kerker/pulay-kerker mixing, "
                                 f"but mixing_type={mixing_type_str}")

    def _validate_kpoint_params(self, params: Any):
        """
        Validate common k-point parameters.

        Checks kspacing and gamma_only for valid values and mutual exclusivity.

        Args:
            params: Parameter object with k-point attributes
        """
        # Validate kspacing
        if hasattr(params, 'kspacing') and params.kspacing is not None:
            if params.kspacing <= 0:
                self._add_error('kspacing', f"kspacing must be > 0, got {params.kspacing}")
            elif params.kspacing > 1.0:
                self._add_warning('kspacing',
                                f"kspacing={params.kspacing} Å⁻¹ is large, k-mesh may be too coarse")
            elif params.kspacing < 0.05:
                self._add_warning('kspacing',
                                f"kspacing={params.kspacing} Å⁻¹ is small, k-mesh may be very dense and expensive")
            else:
                self._add_info('kspacing', f"kspacing={params.kspacing} Å⁻¹ is within typical range")

        # Check mutual exclusivity: gamma_only and kspacing
        if hasattr(params, 'gamma_only') and hasattr(params, 'kspacing'):
            if params.gamma_only and params.kspacing is not None:
                self._add_error('kspacing',
                              "kspacing and gamma_only=True are mutually exclusive")

    def _validate_force_stress_params(self, params: Any):
        """
        Validate force and stress threshold parameters.

        Checks force_thr_ev and stress_thr for valid ranges.

        Args:
            params: Parameter object with force/stress attributes
        """
        # Validate force_thr_ev
        if hasattr(params, 'force_thr_ev') and params.force_thr_ev is not None:
            if params.force_thr_ev <= 0:
                self._add_error('force_thr_ev',
                              f"force_thr_ev must be > 0, got {params.force_thr_ev}")
            elif params.force_thr_ev > 0.1:
                self._add_warning('force_thr_ev',
                                f"force_thr_ev={params.force_thr_ev} eV/Å is large, "
                                "may result in poorly relaxed structure")
            elif params.force_thr_ev < 0.0001:
                self._add_warning('force_thr_ev',
                                f"force_thr_ev={params.force_thr_ev} eV/Å is very tight, "
                                "may be difficult to achieve")
            else:
                self._add_info('force_thr_ev',
                             f"force_thr_ev={params.force_thr_ev} eV/Å is within typical range")

        # Validate stress_thr
        if hasattr(params, 'stress_thr') and params.stress_thr is not None:
            if params.stress_thr <= 0:
                self._add_error('stress_thr', f"stress_thr must be > 0, got {params.stress_thr}")
            elif params.stress_thr > 10:
                self._add_warning('stress_thr',
                                f"stress_thr={params.stress_thr} kBar is large, "
                                "cell may not be well relaxed")
            else:
                self._add_info('stress_thr', f"stress_thr={params.stress_thr} kBar is valid")

    def _validate_output_params(self, params: Any, context: Dict[str, Any]):
        """
        Validate output control parameters.

        Checks out_chg, out_mul, and their dependencies on basis type.

        Args:
            params: Parameter object with output attributes
            context: Context dictionary with basis_type information
        """
        # Validate out_chg
        if hasattr(params, 'out_chg') and params.out_chg is not None:
            if params.out_chg not in [-1, 0, 1]:
                self._add_error('out_chg', f"out_chg must be -1, 0, or 1, got {params.out_chg}")
            else:
                self._add_info('out_chg', f"out_chg={params.out_chg} is valid")

        # Validate out_mul (only works with LCAO)
        if hasattr(params, 'out_mul') and params.out_mul is not None:
            if params.out_mul:
                basis_type = context.get('basis_type', 'pw')
                if basis_type != 'lcao':
                    self._add_warning('out_mul',
                                    f"out_mul only works with LCAO basis, but basis_type={basis_type}")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _validate_positive_float(self, param_name: str, value: Optional[float]):
        """
        Validate that a float parameter is positive.

        Args:
            param_name: Name of the parameter
            value: Value to validate
        """
        if value is not None and value <= 0:
            self._add_error(param_name, f"{param_name} must be > 0, got {value}")

    def _validate_positive_int(self, param_name: str, value: Optional[int]):
        """
        Validate that an int parameter is positive.

        Args:
            param_name: Name of the parameter
            value: Value to validate
        """
        if value is not None and value <= 0:
            self._add_error(param_name, f"{param_name} must be > 0, got {value}")

    def _validate_range(
        self,
        param_name: str,
        value: Optional[float],
        min_val: float,
        max_val: float,
        warn_only: bool = False
    ):
        """
        Validate that a parameter is within a range.

        Args:
            param_name: Name of the parameter
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            warn_only: If True, issue warning instead of error
        """
        if value is not None:
            if value < min_val or value > max_val:
                msg = f"{param_name}={value} outside recommended range [{min_val}, {max_val}]"
                if warn_only:
                    self._add_warning(param_name, msg)
                else:
                    self._add_error(param_name, msg)

    def _add_error(self, parameter: str, message: str):
        """
        Add an error (blocks execution).

        Args:
            parameter: Name of the parameter
            message: Error message
        """
        result = ValidationResult(
            is_valid=False,
            parameter=parameter,
            message=message,
            severity="error"
        )
        self.validation_results.append(result)
        self.errors.append(f"[{parameter}] {message}")

    def _add_warning(self, parameter: str, message: str):
        """
        Add a warning (allows execution).

        Args:
            parameter: Name of the parameter
            message: Warning message
        """
        result = ValidationResult(
            is_valid=True,
            parameter=parameter,
            message=message,
            severity="warning"
        )
        self.validation_results.append(result)
        self.warnings.append(f"[{parameter}] {message}")

    def _add_info(self, parameter: str, message: str):
        """
        Add an info message.

        Args:
            parameter: Name of the parameter
            message: Info message
        """
        result = ValidationResult(
            is_valid=True,
            parameter=parameter,
            message=message,
            severity="info"
        )
        self.validation_results.append(result)
