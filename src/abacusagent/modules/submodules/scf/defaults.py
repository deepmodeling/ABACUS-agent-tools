"""
Default values and inference rules for SCF parameters.

This module implements the inference principle:
- All defaults are explicit and documented
- Inference rules are traceable
- Parameter dependencies are resolved systematically
"""

from typing import Dict, Any, Optional
from copy import deepcopy

from .schema import SCFParameters, SmearingMethod, MixingType
from .audit import SCFAuditLogger


class SCFDefaultsManager:
    """
    Manages default values and inference rules for SCF parameters.

    Design principle: All defaults and inference logic are explicit and documented.
    Each parameter's default value has a clear reasoning, and all inference rules
    are traceable through the audit trail.

    Usage:
        defaults_mgr = SCFDefaultsManager(audit_logger)
        complete_params = defaults_mgr.apply_defaults_and_inferences(params, context)
    """

    def __init__(self, audit_logger: SCFAuditLogger):
        """
        Initialize defaults manager.

        Args:
            audit_logger: Audit logger for tracking parameter provenance
        """
        self.audit = audit_logger

    def apply_defaults_and_inferences(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """
        Apply defaults and inference rules to fill in missing parameters.

        This method processes parameters in dependency order:
        1. Apply basic defaults (no dependencies)
        2. Apply context-dependent defaults
        3. Apply inference rules (depend on other parameters)

        Args:
            params: Partially filled SCF parameters
            context: Context from INPUT file (basis_type, soc, etc.)

        Returns:
            Complete SCF parameters with all values filled
        """
        # Work with a copy to avoid modifying original
        params = deepcopy(params)

        # Apply defaults in dependency order
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params)
        params = self._apply_advanced_defaults(params, context)

        # Apply inference rules (depend on other parameters)
        params = self._infer_from_dependencies(params, context)

        return params

    def _apply_convergence_defaults(self, params: SCFParameters) -> SCFParameters:
        """Apply defaults for convergence parameters."""

        if params.scf_thr is None:
            params.scf_thr = 1e-6
            self.audit.log_default(
                "scf_thr",
                1e-6,
                "Standard convergence threshold for most calculations"
            )

        if params.scf_nmax is None:
            params.scf_nmax = 100
            self.audit.log_default(
                "scf_nmax",
                100,
                "Standard maximum iterations for SCF convergence"
            )

        # ecutwfc is typically inferred from pseudopotential or provided by user
        # Don't set a default here - let it be None if not provided
        if params.ecutwfc is not None:
            self.audit.log_user_input("ecutwfc", params.ecutwfc)

        return params

    def _apply_smearing_defaults(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """Apply defaults for smearing parameters."""

        if params.smearing_method is None:
            params.smearing_method = SmearingMethod.GAUSSIAN
            self.audit.log_default(
                "smearing_method",
                "gaussian",
                "Gaussian smearing is a safe default for most systems"
            )

        if params.smearing_sigma is None:
            # Default depends on system type, but we use a conservative value
            params.smearing_sigma = 0.015  # Ry ≈ 0.2 eV
            self.audit.log_default(
                "smearing_sigma",
                0.015,
                "0.015 Ry (≈0.2 eV) is a reasonable default for semiconductors/insulators"
            )

        return params

    def _apply_mixing_defaults(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """Apply defaults for mixing parameters."""

        if params.mixing_type is None:
            params.mixing_type = MixingType.PULAY
            self.audit.log_default(
                "mixing_type",
                "pulay",
                "Pulay mixing provides good convergence for most systems"
            )

        # mixing_beta default depends on mixing_type
        # This will be handled in inference rules

        if params.mixing_ndim is None:
            # Only set if using pulay/broyden
            mixing_type_str = params.mixing_type.value if isinstance(params.mixing_type, MixingType) else params.mixing_type
            if mixing_type_str in ["pulay", "broyden", "pulay-kerker"]:
                params.mixing_ndim = 8
                self.audit.log_default(
                    "mixing_ndim",
                    8,
                    f"Standard history size for {mixing_type_str} mixing"
                )

        if params.mixing_gg0 is None:
            # Only set if using kerker-based mixing
            mixing_type_str = params.mixing_type.value if isinstance(params.mixing_type, MixingType) else params.mixing_type
            if mixing_type_str in ["kerker", "pulay-kerker"]:
                # Default to 0.0, but user should consider setting > 0 for metals
                params.mixing_gg0 = 0.0
                self.audit.log_default(
                    "mixing_gg0",
                    0.0,
                    "Default Kerker screening (consider 1.0-1.5 for metals)"
                )

        return params

    def _apply_kpoint_defaults(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """Apply defaults for k-point parameters."""

        if params.gamma_only is None:
            params.gamma_only = False
            self.audit.log_default(
                "gamma_only",
                False,
                "Use k-point mesh by default (gamma_only for large supercells)"
            )

        # kspacing is optional - if not provided, use KPT file
        if params.kspacing is not None:
            self.audit.log_user_input("kspacing", params.kspacing)

        return params

    def _apply_output_defaults(self, params: SCFParameters) -> SCFParameters:
        """Apply defaults for output parameters."""

        if params.symmetry is None:
            params.symmetry = True
            self.audit.log_default(
                "symmetry",
                True,
                "Exploit crystal symmetry to reduce computational cost"
            )

        if params.out_chg is None:
            params.out_chg = 0
            self.audit.log_default(
                "out_chg",
                0,
                "Don't output charge density by default (saves disk space)"
            )

        if params.out_mul is None:
            params.out_mul = False
            self.audit.log_default(
                "out_mul",
                False,
                "Don't perform Mulliken analysis by default"
            )

        return params

    def _apply_advanced_defaults(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """Apply defaults for advanced parameters."""

        if params.chg_extrap is None:
            params.chg_extrap = "atomic"
            self.audit.log_default(
                "chg_extrap",
                "atomic",
                "Use atomic charge density extrapolation (standard for SCF)"
            )

        if params.ks_solver is None:
            # Default depends on basis type
            basis_type = context.get("basis_type", "lcao")
            if basis_type == "lcao":
                params.ks_solver = "genelpa"
                self.audit.log_default(
                    "ks_solver",
                    "genelpa",
                    "GENELPA is the default solver for LCAO basis"
                )
            else:
                params.ks_solver = "cg"
                self.audit.log_default(
                    "ks_solver",
                    "cg",
                    "Conjugate gradient is the default solver for PW basis"
                )

        return params

    def _infer_from_dependencies(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """
        Apply inference rules based on parameter dependencies.

        These rules infer parameter values based on other parameters,
        implementing physical and computational best practices.
        """

        # Infer mixing_beta based on mixing_type
        if params.mixing_beta is None:
            mixing_type_str = params.mixing_type.value if isinstance(params.mixing_type, MixingType) else params.mixing_type

            if mixing_type_str == "plain":
                params.mixing_beta = 0.7
                self.audit.log_inferred(
                    "mixing_beta",
                    0.7,
                    "Plain mixing typically uses higher beta (0.7) for reasonable convergence",
                    depends_on=["mixing_type"],
                    inference_rule="plain_mixing_beta"
                )
            elif mixing_type_str in ["pulay", "broyden", "pulay-kerker"]:
                params.mixing_beta = 0.4
                self.audit.log_inferred(
                    "mixing_beta",
                    0.4,
                    f"{mixing_type_str.capitalize()} mixing uses moderate beta (0.4) for stability",
                    depends_on=["mixing_type"],
                    inference_rule="pulay_broyden_mixing_beta"
                )
            elif mixing_type_str == "kerker":
                params.mixing_beta = 0.7
                self.audit.log_inferred(
                    "mixing_beta",
                    0.7,
                    "Kerker mixing uses higher beta (0.7) due to preconditioning",
                    depends_on=["mixing_type"],
                    inference_rule="kerker_mixing_beta"
                )

        # Infer smearing recommendations for metals
        # (This is informational - we don't change user's choice)
        if context.get("is_metallic", False):
            smearing_str = params.smearing_method.value if isinstance(params.smearing_method, SmearingMethod) else params.smearing_method
            if smearing_str == "gaussian":
                self.audit.add_warning(
                    "For metallic systems, consider using smearing_method='mp' (Methfessel-Paxton) "
                    "for better energy accuracy"
                )

        # Infer nspin if not set (from context or default to 1)
        if params.nspin is None:
            nspin_from_context = context.get("nspin", 1)
            params.nspin = nspin_from_context
            if nspin_from_context != 1:
                self.audit.log_dependency(
                    "nspin",
                    nspin_from_context,
                    f"Inherited from INPUT file context",
                    depends_on=["context"]
                )
            else:
                self.audit.log_default(
                    "nspin",
                    1,
                    "Non-spin-polarized calculation (default for non-magnetic systems)"
                )

        return params
