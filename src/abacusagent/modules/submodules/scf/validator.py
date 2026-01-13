"""
Validation logic and dependency rules for SCF parameters.

This module implements the logic-explicit principle:
- All parameter dependencies encoded as explicit rules
- Clear error messages for invalid combinations
- Warnings for suboptimal choices
- Structured validation results
"""

from typing import Dict, List, Tuple, Optional, Literal, Any
from dataclasses import dataclass

from .schema import SCFParameters, MixingType, SmearingMethod


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    """Whether the validation passed"""

    parameter: str
    """Parameter name being validated"""

    message: str
    """Human-readable validation message"""

    severity: Literal["error", "warning", "info"]
    """
    Severity level:
    - error: Blocks execution, must be fixed
    - warning: Allows execution, but may cause issues
    - info: Informational message
    """

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "parameter": self.parameter,
            "message": self.message,
            "severity": self.severity,
        }


class SCFParameterValidator:
    """
    Validates SCF parameters and enforces dependency rules.

    Design principle: All validation logic is explicit and traceable.
    Each validation method checks a specific constraint and produces
    clear error/warning messages.

    Usage:
        validator = SCFParameterValidator()
        is_valid, results = validator.validate_all(params, context)
        if not is_valid:
            # Handle errors
    """

    def __init__(self):
        """Initialize validator with empty result lists."""
        self.validation_results: List[ValidationResult] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

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

        # Range validations
        self._validate_ecutwfc(params.ecutwfc)
        self._validate_scf_thr(params.scf_thr)
        self._validate_scf_nmax(params.scf_nmax)
        self._validate_smearing_sigma(params.smearing_sigma)
        self._validate_mixing_beta(params.mixing_beta)
        self._validate_mixing_ndim(params.mixing_ndim)
        self._validate_mixing_gg0(params.mixing_gg0)
        self._validate_kspacing(params.kspacing)
        self._validate_out_chg(params.out_chg)

        # Dependency validations
        self._validate_mixing_dependencies(params)
        self._validate_smearing_dependencies(params)
        self._validate_kpoint_dependencies(params)
        self._validate_spin_dependencies(params, context)

        # Cross-parameter validations
        self._validate_parameter_compatibility(params, context)

        # Validation passes if there are no errors (warnings are OK)
        is_valid = len(self.errors) == 0
        return is_valid, self.validation_results

    # ========== Range Validations ==========

    def _validate_ecutwfc(self, ecutwfc: Optional[float]):
        """Validate energy cutoff range."""
        if ecutwfc is not None:
            if ecutwfc <= 0:
                self._add_error("ecutwfc", f"ecutwfc must be > 0, got {ecutwfc}")
            elif ecutwfc < 20:
                self._add_warning(
                    "ecutwfc",
                    f"ecutwfc={ecutwfc} Ry is very low, may cause inaccurate results. "
                    "Typical range: 50-150 Ry"
                )
            elif ecutwfc > 200:
                self._add_warning(
                    "ecutwfc",
                    f"ecutwfc={ecutwfc} Ry is very high, may be unnecessarily expensive. "
                    "Typical range: 50-150 Ry"
                )

    def _validate_scf_thr(self, scf_thr: Optional[float]):
        """Validate SCF convergence threshold."""
        if scf_thr is not None:
            if scf_thr <= 0:
                self._add_error("scf_thr", f"scf_thr must be > 0, got {scf_thr}")
            elif scf_thr > 1e-3:
                self._add_warning(
                    "scf_thr",
                    f"scf_thr={scf_thr:.2e} is loose, may cause inaccurate results. "
                    "Typical range: 1e-6 to 1e-9"
                )
            elif scf_thr < 1e-12:
                self._add_warning(
                    "scf_thr",
                    f"scf_thr={scf_thr:.2e} is very tight, may be hard to converge. "
                    "Typical range: 1e-6 to 1e-9"
                )

    def _validate_scf_nmax(self, scf_nmax: Optional[int]):
        """Validate maximum SCF iterations."""
        if scf_nmax is not None:
            if scf_nmax <= 0:
                self._add_error("scf_nmax", f"scf_nmax must be > 0, got {scf_nmax}")
            elif scf_nmax < 20:
                self._add_warning(
                    "scf_nmax",
                    f"scf_nmax={scf_nmax} is low, SCF may not converge. "
                    "Typical range: 50-200"
                )

    def _validate_smearing_sigma(self, smearing_sigma: Optional[float]):
        """Validate smearing width."""
        if smearing_sigma is not None:
            if smearing_sigma <= 0:
                self._add_error(
                    "smearing_sigma",
                    f"smearing_sigma must be > 0, got {smearing_sigma}"
                )
            elif smearing_sigma > 0.1:
                # 0.1 Ry ≈ 1.4 eV
                self._add_warning(
                    "smearing_sigma",
                    f"smearing_sigma={smearing_sigma} Ry (≈{smearing_sigma*13.6:.1f} eV) is large, "
                    "may over-smear electronic structure. Typical range: 0.01-0.05 Ry"
                )
            elif smearing_sigma < 0.001:
                self._add_warning(
                    "smearing_sigma",
                    f"smearing_sigma={smearing_sigma} Ry is very small, may cause poor SCF convergence"
                )

    def _validate_mixing_beta(self, mixing_beta: Optional[float]):
        """Validate mixing parameter."""
        if mixing_beta is not None:
            if mixing_beta <= 0 or mixing_beta > 1:
                self._add_error(
                    "mixing_beta",
                    f"mixing_beta must be in (0, 1], got {mixing_beta}"
                )
            elif mixing_beta > 0.8:
                self._add_warning(
                    "mixing_beta",
                    f"mixing_beta={mixing_beta} is high, may cause SCF instability. "
                    "Consider reducing to 0.3-0.7"
                )
            elif mixing_beta < 0.1:
                self._add_warning(
                    "mixing_beta",
                    f"mixing_beta={mixing_beta} is very low, SCF convergence may be slow"
                )

    def _validate_mixing_ndim(self, mixing_ndim: Optional[int]):
        """Validate mixing dimension."""
        if mixing_ndim is not None:
            if mixing_ndim <= 0:
                self._add_error(
                    "mixing_ndim",
                    f"mixing_ndim must be > 0, got {mixing_ndim}"
                )
            elif mixing_ndim > 20:
                self._add_warning(
                    "mixing_ndim",
                    f"mixing_ndim={mixing_ndim} is large, may use excessive memory. "
                    "Typical range: 4-20"
                )
            elif mixing_ndim < 4:
                self._add_warning(
                    "mixing_ndim",
                    f"mixing_ndim={mixing_ndim} is small, may reduce mixing effectiveness"
                )

    def _validate_mixing_gg0(self, mixing_gg0: Optional[float]):
        """Validate Kerker screening parameter."""
        if mixing_gg0 is not None:
            if mixing_gg0 < 0:
                self._add_error(
                    "mixing_gg0",
                    f"mixing_gg0 must be ≥ 0, got {mixing_gg0}"
                )

    def _validate_kspacing(self, kspacing: Optional[float]):
        """Validate k-point spacing."""
        if kspacing is not None:
            if kspacing <= 0:
                self._add_error("kspacing", f"kspacing must be > 0, got {kspacing}")
            elif kspacing > 1.0:
                self._add_warning(
                    "kspacing",
                    f"kspacing={kspacing} is large, k-mesh may be too coarse. "
                    "Typical range: 0.1-0.5"
                )
            elif kspacing < 0.05:
                self._add_warning(
                    "kspacing",
                    f"kspacing={kspacing} is very small, k-mesh may be unnecessarily dense"
                )

    def _validate_out_chg(self, out_chg: Optional[int]):
        """Validate charge output parameter."""
        if out_chg is not None:
            if out_chg not in [-1, 0, 1]:
                self._add_error(
                    "out_chg",
                    f"out_chg must be -1, 0, or 1, got {out_chg}"
                )

    # ========== Dependency Validations ==========

    def _validate_mixing_dependencies(self, params: SCFParameters):
        """
        Validate mixing parameter dependencies.

        Rules:
        1. mixing_ndim only applies to pulay/broyden mixing
        2. mixing_gg0 only applies to kerker-based mixing
        3. mixing_beta defaults depend on mixing_type
        """
        if params.mixing_type is None:
            return

        mixing_type_str = params.mixing_type.value if isinstance(params.mixing_type, MixingType) else params.mixing_type

        # Check mixing_ndim applicability
        if mixing_type_str in ["pulay", "broyden", "pulay-kerker"]:
            if params.mixing_ndim is None:
                self._add_info(
                    "mixing_ndim",
                    f"mixing_type={mixing_type_str} uses mixing_ndim, will use default=8"
                )
        else:
            if params.mixing_ndim is not None:
                self._add_warning(
                    "mixing_ndim",
                    f"mixing_ndim is ignored for mixing_type={mixing_type_str}. "
                    "Only applies to pulay/broyden/pulay-kerker"
                )

        # Check mixing_gg0 applicability
        if mixing_type_str in ["kerker", "pulay-kerker"]:
            if params.mixing_gg0 is None or params.mixing_gg0 == 0:
                self._add_info(
                    "mixing_gg0",
                    f"mixing_type={mixing_type_str} benefits from mixing_gg0 > 0 "
                    "(e.g., 1.0-1.5 for metals)"
                )
        else:
            if params.mixing_gg0 is not None and params.mixing_gg0 > 0:
                self._add_warning(
                    "mixing_gg0",
                    f"mixing_gg0 is ignored for mixing_type={mixing_type_str}. "
                    "Only applies to kerker/pulay-kerker"
                )

    def _validate_smearing_dependencies(self, params: SCFParameters):
        """
        Validate smearing parameter dependencies.

        Rules:
        1. smearing_sigma is required if smearing_method != fixed
        2. Recommend appropriate methods for different systems
        """
        if params.smearing_method is None:
            return

        smearing_str = params.smearing_method.value if isinstance(params.smearing_method, SmearingMethod) else params.smearing_method

        if smearing_str != "fixed":
            if params.smearing_sigma is None:
                self._add_info(
                    "smearing_sigma",
                    f"smearing_method={smearing_str} requires smearing_sigma, will use default"
                )

    def _validate_kpoint_dependencies(self, params: SCFParameters):
        """
        Validate k-point parameter dependencies.

        Rules:
        1. gamma_only and kspacing are mutually exclusive
        """
        if params.gamma_only and params.kspacing is not None:
            self._add_error(
                "kspacing",
                "kspacing and gamma_only=True are mutually exclusive. "
                "Use either gamma_only for single k-point or kspacing for automatic mesh"
            )

    def _validate_spin_dependencies(self, params: SCFParameters, context: Dict[str, Any]):
        """
        Validate spin-related dependencies.

        Rules:
        1. If soc=True (from context), nspin must be 4
        2. If nspin=4, must use non-collinear calculation
        """
        soc = context.get("soc", False)
        nspin = params.nspin if params.nspin is not None else context.get("nspin", 1)

        if soc and nspin != 4:
            self._add_error(
                "nspin",
                f"Spin-orbit coupling (soc=True) requires nspin=4, got nspin={nspin}. "
                "Set nspin=4 or disable SOC"
            )

    def _validate_parameter_compatibility(self, params: SCFParameters, context: Dict[str, Any]):
        """
        Validate cross-parameter compatibility.

        Rules:
        1. PW basis requires ecutwfc
        2. LCAO basis may need orbital files
        3. out_mul only works with LCAO
        """
        basis_type = context.get("basis_type", "lcao")

        # Check ecutwfc for PW basis
        if basis_type == "pw":
            if params.ecutwfc is None:
                self._add_info(
                    "ecutwfc",
                    "PW basis requires ecutwfc, will use default or infer from pseudopotential"
                )

        # Check out_mul for LCAO
        if params.out_mul and basis_type != "lcao":
            self._add_warning(
                "out_mul",
                f"Mulliken analysis (out_mul=True) only available for LCAO basis, "
                f"got basis_type={basis_type}"
            )

    # ========== Helper Methods ==========

    def _add_error(self, parameter: str, message: str):
        """Add an error (blocks execution)."""
        result = ValidationResult(
            is_valid=False,
            parameter=parameter,
            message=message,
            severity="error"
        )
        self.validation_results.append(result)
        self.errors.append(f"[{parameter}] {message}")

    def _add_warning(self, parameter: str, message: str):
        """Add a warning (allows execution)."""
        result = ValidationResult(
            is_valid=True,
            parameter=parameter,
            message=message,
            severity="warning"
        )
        self.validation_results.append(result)
        self.warnings.append(f"[{parameter}] {message}")

    def _add_info(self, parameter: str, message: str):
        """Add an info message."""
        result = ValidationResult(
            is_valid=True,
            parameter=parameter,
            message=message,
            severity="info"
        )
        self.validation_results.append(result)
