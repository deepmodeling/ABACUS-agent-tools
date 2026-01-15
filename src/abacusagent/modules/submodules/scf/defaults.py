"""
Default values and inference rules for SCF parameters.

This module implements the inference principle:
- All defaults are explicit and documented
- Inference rules are traceable
- Parameter dependencies are resolved systematically
- Inherits common defaults from the shared framework
"""

from typing import Dict, Any
from copy import deepcopy

from ..common import BaseDefaultsManager, INFERENCE_RULES
from .schema import SCFParameters, SmearingMethod, MixingType
from .audit import SCFAuditLogger


class SCFDefaultsManager(BaseDefaultsManager):
    """
    Manages default values and inference rules for SCF parameters.

    Inherits common default application methods from BaseDefaultsManager:
    - _apply_convergence_defaults(): scf_thr, scf_nmax
    - _apply_smearing_defaults(): smearing_method, smearing_sigma
    - _apply_mixing_defaults(): mixing_type, mixing_ndim
    - _apply_kpoint_defaults(): gamma_only
    - _apply_output_defaults(): symmetry, out_chg, out_mul
    - _infer_mixing_beta(): Infer from mixing_type
    - _infer_ks_solver(): Infer from basis_type

    Adds SCF-specific defaults:
    - chg_extrap: Charge extrapolation method
    - nspin: Number of spin channels

    Design principle: All defaults and inference logic are explicit and documented.

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
        super().__init__(audit_logger)

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

        # Apply common defaults (inherited from base)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)

        # Apply SCF-specific defaults
        params = self._apply_scf_advanced_defaults(params, context)

        # Apply inference rules (depend on other parameters)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        params = self._infer_scf_specific(params, context)

        return params

    # ========== SCF-Specific Default Application ==========

    def _apply_scf_advanced_defaults(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """Apply defaults for SCF-specific advanced parameters."""

        # chg_extrap default
        if params.chg_extrap is None:
            params.chg_extrap = "atomic"
            self.audit.log_default(
                "chg_extrap",
                "atomic",
                "Use atomic charge density extrapolation (standard for SCF)"
            )

        return params

    # ========== SCF-Specific Inference Rules ==========

    def _infer_scf_specific(
        self,
        params: SCFParameters,
        context: Dict[str, Any]
    ) -> SCFParameters:
        """
        Apply SCF-specific inference rules.

        Infers:
        - nspin: From context or default to 1
        - Smearing recommendations for metals (warning only)
        """

        # Infer nspin if not set (from context or default to 1)
        if params.nspin is None:
            nspin_from_context = context.get("nspin", 1)
            params.nspin = nspin_from_context
            if nspin_from_context != 1:
                self.audit.log_dependency(
                    "nspin",
                    nspin_from_context,
                    "Inherited from INPUT file context",
                    depends_on=["context"]
                )
            else:
                self.audit.log_default(
                    "nspin",
                    1,
                    "Non-spin-polarized calculation (default for non-magnetic systems)"
                )

        # Infer smearing recommendations for metals (informational only)
        if context.get("is_metallic", False):
            if params.smearing_method is not None:
                smearing_str = self._get_enum_value(params.smearing_method)
                if smearing_str == "gaussian":
                    self.audit.add_warning(
                        "For metallic systems, consider using smearing_method='mp' (Methfessel-Paxton) "
                        "for better energy accuracy"
                    )

        return params


# ============================================================================
# RE-EXPORT FOR BACKWARD COMPATIBILITY
# ============================================================================

__all__ = [
    "SCFDefaultsManager",
]
