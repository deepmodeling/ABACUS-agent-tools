"""
Base defaults manager providing common default application and inference rules.

This module provides the BaseDefaultsManager class that all module-specific
defaults managers inherit from. It includes default values and inference rules
for common parameters like convergence, smearing, mixing, and k-points.
"""

from typing import Dict, Any
from copy import deepcopy
from .base_schema import BaseParameters
from .base_audit import BaseAuditLogger
from .shared_parameters import SmearingMethod, MixingType


# ============================================================================
# Named Inference Rules
# ============================================================================

INFERENCE_RULES = {
    # Mixing inference rules
    "pulay_mixing_beta_default": "Pulay mixing uses moderate beta (0.4) for stability",
    "broyden_mixing_beta_default": "Broyden mixing uses moderate beta (0.4) for stability",
    "plain_mixing_beta_default": "Plain mixing uses higher beta (0.7) for faster convergence",
    "kerker_mixing_beta_default": "Kerker mixing uses higher beta (0.7) with screening",

    # Solver inference rules
    "lcao_ks_solver_default": "LCAO basis uses genelpa solver for efficiency",
    "pw_ks_solver_default": "PW basis uses cg solver as standard",

    # Smearing inference rules
    "metal_smearing_suggestion": "Metallic systems benefit from Methfessel-Paxton smearing",
    "semiconductor_smearing_default": "Gaussian smearing is safe for semiconductors/insulators",

    # Convergence inference rules
    "tight_convergence_mixing": "Tight convergence (scf_thr < 1e-8) requires lower mixing_beta",
}


class BaseDefaultsManager:
    """
    Base defaults manager providing common default application methods.

    Module-specific managers inherit from this and add their own rules.
    This ensures consistent defaults for common parameters across all modules.

    Attributes:
        audit: Audit logger for tracking parameter provenance
    """

    def __init__(self, audit_logger: BaseAuditLogger):
        """
        Initialize defaults manager.

        Args:
            audit_logger: Audit logger for tracking parameter provenance
        """
        self.audit = audit_logger

    def apply_defaults_and_inferences(
        self,
        params: BaseParameters,
        context: Dict[str, Any]
    ) -> BaseParameters:
        """
        Apply defaults and inference rules to fill in missing parameters.

        Must be implemented by subclasses to define the full default workflow.

        Args:
            params: Parameter object with user-provided values
            context: Context dictionary with additional information
                (e.g., basis_type, soc, existing INPUT parameters)

        Returns:
            Parameter object with defaults and inferences applied
        """
        raise NotImplementedError("Subclasses must implement apply_defaults_and_inferences()")

    # ========================================================================
    # Common Default Application Methods
    # ========================================================================

    def _apply_convergence_defaults(self, params: Any) -> Any:
        """
        Apply defaults for common convergence parameters.

        Sets standard defaults for scf_thr and scf_nmax if not provided.

        Args:
            params: Parameter object with convergence attributes

        Returns:
            Parameter object with convergence defaults applied
        """
        if hasattr(params, 'scf_thr') and params.scf_thr is None:
            params.scf_thr = 1e-6
            self.audit.log_default(
                "scf_thr",
                1e-6,
                "Standard convergence threshold for most calculations"
            )

        if hasattr(params, 'scf_nmax') and params.scf_nmax is None:
            params.scf_nmax = 100
            self.audit.log_default(
                "scf_nmax",
                100,
                "Standard maximum iterations for SCF convergence"
            )

        return params

    def _apply_smearing_defaults(self, params: Any, context: Dict[str, Any]) -> Any:
        """
        Apply defaults for common smearing parameters.

        Sets Gaussian smearing as default with sigma = 0.015 Ry (≈0.2 eV).

        Args:
            params: Parameter object with smearing attributes
            context: Context dictionary (not currently used)

        Returns:
            Parameter object with smearing defaults applied
        """
        if hasattr(params, 'smearing_method') and params.smearing_method is None:
            params.smearing_method = SmearingMethod.GAUSSIAN
            self.audit.log_default(
                "smearing_method",
                "gaussian",
                "Gaussian smearing is a safe default for most systems"
            )

        if hasattr(params, 'smearing_sigma') and params.smearing_sigma is None:
            params.smearing_sigma = 0.015  # Ry ≈ 0.2 eV
            self.audit.log_default(
                "smearing_sigma",
                0.015,
                "0.015 Ry (≈0.2 eV) is a reasonable default for semiconductors/insulators"
            )

        return params

    def _apply_mixing_defaults(self, params: Any, context: Dict[str, Any]) -> Any:
        """
        Apply defaults for common mixing parameters.

        Sets Pulay mixing as default with appropriate ndim.

        Args:
            params: Parameter object with mixing attributes
            context: Context dictionary (not currently used)

        Returns:
            Parameter object with mixing defaults applied
        """
        if hasattr(params, 'mixing_type') and params.mixing_type is None:
            params.mixing_type = MixingType.PULAY
            self.audit.log_default(
                "mixing_type",
                "pulay",
                "Pulay mixing provides good convergence for most systems"
            )

        # mixing_ndim default (only for pulay/broyden)
        if hasattr(params, 'mixing_ndim') and params.mixing_ndim is None:
            if hasattr(params, 'mixing_type') and params.mixing_type is not None:
                mixing_type_str = params.mixing_type.value if hasattr(params.mixing_type, 'value') else str(params.mixing_type)
                if mixing_type_str in ["pulay", "broyden", "pulay-kerker"]:
                    params.mixing_ndim = 8
                    self.audit.log_default(
                        "mixing_ndim",
                        8,
                        f"Standard history size for {mixing_type_str} mixing"
                    )

        return params

    def _apply_kpoint_defaults(self, params: Any, context: Dict[str, Any]) -> Any:
        """
        Apply defaults for common k-point parameters.

        Sets gamma_only = False as default.

        Args:
            params: Parameter object with k-point attributes
            context: Context dictionary (not currently used)

        Returns:
            Parameter object with k-point defaults applied
        """
        if hasattr(params, 'gamma_only') and params.gamma_only is None:
            params.gamma_only = False
            self.audit.log_default(
                "gamma_only",
                False,
                "Use k-point mesh for better accuracy (not just Gamma point)"
            )

        return params

    def _apply_output_defaults(self, params: Any, context: Dict[str, Any]) -> Any:
        """
        Apply defaults for common output parameters.

        Sets standard defaults for symmetry, out_chg, and out_mul.

        Args:
            params: Parameter object with output attributes
            context: Context dictionary (not currently used)

        Returns:
            Parameter object with output defaults applied
        """
        if hasattr(params, 'symmetry') and params.symmetry is None:
            params.symmetry = True
            self.audit.log_default(
                "symmetry",
                True,
                "Use crystal symmetry to reduce k-points and speed up calculation"
            )

        if hasattr(params, 'out_chg') and params.out_chg is None:
            params.out_chg = 0
            self.audit.log_default(
                "out_chg",
                0,
                "Do not output charge density by default (saves disk space)"
            )

        if hasattr(params, 'out_mul') and params.out_mul is None:
            params.out_mul = False
            self.audit.log_default(
                "out_mul",
                False,
                "Do not output Mulliken analysis by default"
            )

        return params

    # ========================================================================
    # Common Inference Rules
    # ========================================================================

    def _infer_mixing_beta(self, params: Any) -> Any:
        """
        Infer mixing_beta from mixing_type if not provided.

        Different mixing types have different optimal beta values:
        - plain/kerker: 0.7 (higher for faster convergence)
        - pulay/broyden/pulay-kerker: 0.4 (lower for stability)

        Args:
            params: Parameter object with mixing attributes

        Returns:
            Parameter object with mixing_beta inferred if needed
        """
        if hasattr(params, 'mixing_beta') and params.mixing_beta is None:
            if hasattr(params, 'mixing_type') and params.mixing_type is not None:
                mixing_type_str = params.mixing_type.value if hasattr(params.mixing_type, 'value') else str(params.mixing_type)

                if mixing_type_str == "plain":
                    params.mixing_beta = 0.7
                    self.audit.log_inferred(
                        "mixing_beta",
                        0.7,
                        INFERENCE_RULES["plain_mixing_beta_default"],
                        depends_on=["mixing_type"],
                        inference_rule="plain_mixing_beta_default"
                    )
                elif mixing_type_str == "kerker":
                    params.mixing_beta = 0.7
                    self.audit.log_inferred(
                        "mixing_beta",
                        0.7,
                        INFERENCE_RULES["kerker_mixing_beta_default"],
                        depends_on=["mixing_type"],
                        inference_rule="kerker_mixing_beta_default"
                    )
                elif mixing_type_str in ["pulay", "pulay-kerker"]:
                    params.mixing_beta = 0.4
                    self.audit.log_inferred(
                        "mixing_beta",
                        0.4,
                        INFERENCE_RULES["pulay_mixing_beta_default"],
                        depends_on=["mixing_type"],
                        inference_rule="pulay_mixing_beta_default"
                    )
                elif mixing_type_str == "broyden":
                    params.mixing_beta = 0.4
                    self.audit.log_inferred(
                        "mixing_beta",
                        0.4,
                        INFERENCE_RULES["broyden_mixing_beta_default"],
                        depends_on=["mixing_type"],
                        inference_rule="broyden_mixing_beta_default"
                    )

        return params

    def _infer_ks_solver(self, params: Any, context: Dict[str, Any]) -> Any:
        """
        Infer ks_solver from basis_type if not provided.

        Different basis types have different optimal solvers:
        - lcao: genelpa (efficient for LCAO)
        - pw: cg (standard for plane waves)

        Args:
            params: Parameter object with ks_solver attribute
            context: Context dictionary with basis_type information

        Returns:
            Parameter object with ks_solver inferred if needed
        """
        if hasattr(params, 'ks_solver') and params.ks_solver is None:
            basis_type = context.get('basis_type', 'pw')

            if basis_type == 'lcao':
                params.ks_solver = 'genelpa'
                self.audit.log_inferred(
                    "ks_solver",
                    "genelpa",
                    INFERENCE_RULES["lcao_ks_solver_default"],
                    depends_on=["basis_type"],
                    inference_rule="lcao_ks_solver_default"
                )
            else:  # pw or lcao_in_pw
                params.ks_solver = 'cg'
                self.audit.log_inferred(
                    "ks_solver",
                    "cg",
                    INFERENCE_RULES["pw_ks_solver_default"],
                    depends_on=["basis_type"],
                    inference_rule="pw_ks_solver_default"
                )

        return params

    def _adjust_mixing_for_tight_convergence(self, params: Any) -> Any:
        """
        Adjust mixing_beta for tight convergence if needed.

        If scf_thr is very tight (< 1e-8) and mixing_beta is high (> 0.5),
        suggest lowering mixing_beta for better stability.

        Args:
            params: Parameter object with convergence and mixing attributes

        Returns:
            Parameter object (may add warning to audit)
        """
        if hasattr(params, 'scf_thr') and hasattr(params, 'mixing_beta'):
            if params.scf_thr is not None and params.mixing_beta is not None:
                if params.scf_thr < 1e-8 and params.mixing_beta > 0.5:
                    self.audit.add_warning(
                        f"Tight convergence (scf_thr={params.scf_thr}) with high mixing_beta "
                        f"({params.mixing_beta}) may cause instability. "
                        "Consider using mixing_beta ≤ 0.4"
                    )

        return params

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_enum_value(self, enum_or_str: Any) -> str:
        """
        Get string value from enum or string.

        Args:
            enum_or_str: Enum object or string

        Returns:
            String value
        """
        if hasattr(enum_or_str, 'value'):
            return enum_or_str.value
        return str(enum_or_str)
