"""
Validation logic and dependency rules for SCF parameters.

This module implements the logic-explicit principle:
- All parameter dependencies encoded as explicit rules
- Clear error messages for invalid combinations
- Warnings for suboptimal choices
- Structured validation results
- Inherits common validation from the shared framework
"""

from typing import Dict, Tuple, List, Any
from ..common import BaseParameterValidator, ValidationResult
from .schema import SCFParameters, MixingType, SmearingMethod


class SCFParameterValidator(BaseParameterValidator):
    """
    Validates SCF parameters and enforces dependency rules.

    Inherits common validation methods from BaseParameterValidator:
    - _validate_convergence_params(): Validate scf_thr, scf_nmax, ecutwfc
    - _validate_smearing_params(): Validate smearing_method, smearing_sigma
    - _validate_mixing_params(): Validate mixing_type, mixing_beta, mixing_ndim, mixing_gg0
    - _validate_kpoint_params(): Validate kspacing, gamma_only
    - _validate_output_params(): Validate out_chg, out_mul

    Adds SCF-specific validation:
    - Spin dependencies (nspin, soc)
    - Advanced parameter validation (chg_extrap, ks_solver)
    - Parameter compatibility checks

    Design principle: All validation logic is explicit and traceable.

    Usage:
        validator = SCFParameterValidator()
        is_valid, results = validator.validate_all(params, context)
        if not is_valid:
            # Handle errors
    """

    def validate_all(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.

        Args:
            params: SCF parameters to validate
            context: Additional context from INPUT file (basis_type, soc, etc.)

        Returns:
            Tuple of (is_valid, validation_results)
            - is_valid: True if no errors (warnings are OK)
            - validation_results: List of all validation results
        """
        # Reset state
        self.validation_results = []
        self.warnings = []
        self.errors = []

        # Common validations (inherited from base)
        self._validate_convergence_params(params)
        self._validate_smearing_params(params)
        self._validate_mixing_params(params)
        self._validate_kpoint_params(params)
        self._validate_output_params(params, context)

        # SCF-specific validations
        self._validate_spin_dependencies(params, context)
        self._validate_advanced_params(params, context)

        # Validation passes if there are no errors (warnings are OK)
        is_valid = len(self.errors) == 0
        return is_valid, self.validation_results

    # ========== SCF-Specific Validations ==========

    def _validate_spin_dependencies(self, params: SCFParameters, context: Dict[str, Any]):
        """
        Validate spin-related dependencies.

        Rules:
        1. If soc=True (from context), nspin must be 4
        2. If nspin=4, must use non-collinear calculation
        """
        soc = context.get('soc', False)
        nspin = params.nspin

        if soc and nspin is not None and nspin != 4:
            self._add_error(
                "nspin",
                f"Spin-orbit coupling (soc=True) requires nspin=4, got nspin={nspin}"
            )

        if nspin == 4 and not soc:
            self._add_warning(
                "nspin",
                "nspin=4 typically requires spin-orbit coupling (soc=True)"
            )

    def _validate_advanced_params(self, params: SCFParameters, context: Dict[str, Any]):
        """
        Validate advanced SCF parameters.

        Validates:
        - chg_extrap: Charge extrapolation method
        - ks_solver: Kohn-Sham solver compatibility with basis type
        """
        # Validate ks_solver compatibility with basis_type
        if params.ks_solver is not None:
            basis_type = context.get('basis_type', 'pw')

            # Check if solver is appropriate for basis type
            if basis_type == 'lcao':
                if params.ks_solver in ['cg', 'dav', 'bpcg']:
                    self._add_warning(
                        "ks_solver",
                        f"ks_solver={params.ks_solver} is typically used with PW basis, "
                        f"but basis_type={basis_type}. Consider using genelpa for LCAO."
                    )
            elif basis_type == 'pw':
                if params.ks_solver in ['genelpa', 'scalapack_gvx']:
                    self._add_warning(
                        "ks_solver",
                        f"ks_solver={params.ks_solver} is typically used with LCAO basis, "
                        f"but basis_type={basis_type}. Consider using cg or dav for PW."
                    )

        # chg_extrap validation (just check it's a valid value, no complex rules)
        if params.chg_extrap is not None:
            valid_chg_extrap = ["none", "atomic", "first-order", "second-order"]
            if params.chg_extrap not in valid_chg_extrap:
                self._add_error(
                    "chg_extrap",
                    f"chg_extrap must be one of {valid_chg_extrap}, got {params.chg_extrap}"
                )


# ============================================================================
# RE-EXPORT COMMON TYPES FOR BACKWARD COMPATIBILITY
# ============================================================================

__all__ = [
    "SCFParameterValidator",
    "ValidationResult",
]
