"""
Validation logic and dependency rules for relax parameters.

This module implements the logic-explicit principle:
- All parameter dependencies encoded as explicit rules
- Clear error messages for invalid combinations
- Warnings for suboptimal choices
- Structured validation results
- Inherits common validation from the shared framework
"""

from typing import Dict, Tuple, List, Any
from ..common import BaseParameterValidator, ValidationResult
from .schema import RelaxParameters


class RelaxParameterValidator(BaseParameterValidator):
    """
    Validates relax parameters and enforces dependency rules.

    Inherits common validation methods from BaseParameterValidator:
    - _validate_convergence_params(): Validate scf_thr, scf_nmax, ecutwfc
    - _validate_smearing_params(): Validate smearing_method, smearing_sigma
    - _validate_mixing_params(): Validate mixing_type, mixing_beta, mixing_ndim, mixing_gg0
    - _validate_kpoint_params(): Validate kspacing, gamma_only
    - _validate_force_stress_params(): Validate force_thr_ev, stress_thr
    - _validate_output_params(): Validate out_chg, out_mul

    Adds relax-specific validation:
    - Relaxation control parameters (relax_nmax, relax_method, relax_new)
    - Cell relaxation dependencies (relax_cell, fixed_axes, stress_thr)

    Design principle: All validation logic is explicit and traceable.

    Usage:
        validator = RelaxParameterValidator()
        is_valid, results = validator.validate_all(params, context)
        if not is_valid:
            # Handle errors
    """

    def validate_all(
        self,
        params: RelaxParameters,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.

        Args:
            params: Relax parameters to validate
            context: Additional context from INPUT file (basis_type, etc.)

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
        self._validate_force_stress_params(params)
        self._validate_output_params(params, context)

        # Relax-specific validations
        self._validate_relax_control_params(params)
        self._validate_cell_relax_dependencies(params)

        # Validation passes if there are no errors (warnings are OK)
        is_valid = len(self.errors) == 0
        return is_valid, self.validation_results

    # ========== Relax-Specific Validations ==========

    def _validate_relax_control_params(self, params: RelaxParameters):
        """
        Validate relaxation control parameters.

        Validates:
        - relax_nmax: Maximum relaxation steps
        - relax_method: Relaxation algorithm
        - relax_new: New CG implementation flag
        """
        # Validate relax_nmax
        if params.relax_nmax is not None:
            if params.relax_nmax <= 0:
                self._add_error(
                    "relax_nmax",
                    f"relax_nmax must be > 0, got {params.relax_nmax}"
                )
            elif params.relax_nmax < 20:
                self._add_warning(
                    "relax_nmax",
                    f"relax_nmax={params.relax_nmax} is low, relaxation may not converge"
                )
            else:
                self._add_info(
                    "relax_nmax",
                    f"relax_nmax={params.relax_nmax} is sufficient"
                )

        # Validate relax_method
        if params.relax_method is not None:
            valid_methods = ["cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire"]
            if params.relax_method not in valid_methods:
                self._add_error(
                    "relax_method",
                    f"relax_method must be one of {valid_methods}, got {params.relax_method}"
                )

        # Validate relax_new (only relevant for CG method)
        if params.relax_new is not None and params.relax_method is not None:
            if params.relax_method != "cg" and params.relax_new:
                self._add_info(
                    "relax_new",
                    f"relax_new is only used with relax_method='cg', "
                    f"but relax_method={params.relax_method}"
                )

    def _validate_cell_relax_dependencies(self, params: RelaxParameters):
        """
        Validate cell relaxation parameter dependencies.

        Rules:
        1. stress_thr is only relevant when relax_cell=True
        2. fixed_axes is only relevant when relax_cell=True
        3. If relax_cell=True, stress_thr should be provided
        """
        # Check stress_thr dependency on relax_cell
        if params.stress_thr is not None:
            if params.relax_cell is False:
                self._add_info(
                    "stress_thr",
                    "stress_thr is only used when relax_cell=True (cell-relax)"
                )
            elif params.relax_cell is None:
                self._add_info(
                    "stress_thr",
                    "stress_thr provided, assuming cell-relax calculation"
                )

        # Check fixed_axes dependency on relax_cell
        if params.fixed_axes is not None and params.fixed_axes != "None":
            if params.relax_cell is False:
                self._add_warning(
                    "fixed_axes",
                    f"fixed_axes={params.fixed_axes} is only used when relax_cell=True"
                )
            elif params.relax_cell is None:
                self._add_info(
                    "fixed_axes",
                    f"fixed_axes={params.fixed_axes} provided, assuming cell-relax calculation"
                )

        # If relax_cell=True, recommend providing stress_thr
        if params.relax_cell is True:
            if params.stress_thr is None:
                self._add_info(
                    "stress_thr",
                    "For cell-relax, consider providing stress_thr (default will be used)"
                )


__all__ = [
    "RelaxParameterValidator",
    "ValidationResult",
]
