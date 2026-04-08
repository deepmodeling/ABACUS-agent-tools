"""
Default values and inference rules for relax parameters.

This module implements the inference principle:
- All defaults are explicit and documented
- Inference rules are traceable
- Parameter dependencies are resolved systematically
- Inherits common defaults from the shared framework
"""

from typing import Dict, Any
from copy import deepcopy

from ..common import BaseDefaultsManager, INFERENCE_RULES
from .schema import RelaxParameters
from .audit import RelaxAuditLogger


class RelaxDefaultsManager(BaseDefaultsManager):
    """
    Manages default values and inference rules for relax parameters.

    Inherits common default application methods from BaseDefaultsManager:
    - _apply_convergence_defaults(): scf_thr, scf_nmax
    - _apply_smearing_defaults(): smearing_method, smearing_sigma
    - _apply_mixing_defaults(): mixing_type, mixing_ndim
    - _apply_kpoint_defaults(): gamma_only
    - _apply_output_defaults(): symmetry, out_chg, out_mul
    - _infer_mixing_beta(): Infer from mixing_type
    - _infer_ks_solver(): Infer from basis_type

    Adds relax-specific defaults:
    - force_thr_ev: Force convergence threshold
    - stress_thr: Stress convergence threshold
    - relax_nmax: Maximum relaxation steps
    - relax_method: Relaxation algorithm
    - relax_new: Use new CG implementation
    - relax_cell: Whether to relax cell

    Design principle: All defaults and inference logic are explicit and documented.

    Usage:
        defaults_mgr = RelaxDefaultsManager(audit_logger)
        complete_params = defaults_mgr.apply_defaults_and_inferences(params, context)
    """

    def __init__(self, audit_logger: RelaxAuditLogger):
        """
        Initialize defaults manager.

        Args:
            audit_logger: Audit logger for tracking parameter provenance
        """
        super().__init__(audit_logger)

    def apply_defaults_and_inferences(
        self,
        params: RelaxParameters,
        context: Dict[str, Any]
    ) -> RelaxParameters:
        """
        Apply defaults and inference rules to fill in missing parameters.

        This method processes parameters in dependency order:
        1. Apply basic defaults (no dependencies)
        2. Apply context-dependent defaults
        3. Apply inference rules (depend on other parameters)

        Args:
            params: Partially filled relax parameters
            context: Context from INPUT file (basis_type, etc.)

        Returns:
            Complete relax parameters with all values filled
        """
        # Work with a copy to avoid modifying original
        params = deepcopy(params)

        # Apply common defaults (inherited from base)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)

        # Apply relax-specific defaults
        params = self._apply_force_stress_defaults(params)
        params = self._apply_relax_control_defaults(params)

        # Apply inference rules (depend on other parameters)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        params = self._infer_relax_specific(params, context)

        return params

    # ========== Relax-Specific Default Application ==========

    def _apply_force_stress_defaults(self, params: RelaxParameters) -> RelaxParameters:
        """Apply defaults for force and stress thresholds."""

        if params.force_thr_ev is None:
            params.force_thr_ev = 0.01
            self.audit.log_default(
                "force_thr_ev",
                0.01,
                "Standard force convergence threshold (0.01 eV/Å)"
            )

        if params.stress_thr is None and params.relax_cell:
            params.stress_thr = 1.0
            self.audit.log_default(
                "stress_thr",
                1.0,
                "Standard stress convergence threshold for cell-relax (1.0 kBar)"
            )

        return params

    def _apply_relax_control_defaults(self, params: RelaxParameters) -> RelaxParameters:
        """Apply defaults for relaxation control parameters."""

        if params.relax_nmax is None:
            params.relax_nmax = 100
            self.audit.log_default(
                "relax_nmax",
                100,
                "Standard maximum relaxation steps"
            )

        if params.relax_method is None:
            params.relax_method = "cg"
            self.audit.log_default(
                "relax_method",
                "cg",
                "Conjugate gradient is reliable for most systems"
            )

        if params.relax_new is None and params.relax_method == "cg":
            params.relax_new = True
            self.audit.log_default(
                "relax_new",
                True,
                "Use new CG implementation (recommended)"
            )

        if params.relax_cell is None:
            params.relax_cell = False
            self.audit.log_default(
                "relax_cell",
                False,
                "Relax atomic positions only (not cell parameters)"
            )

        if params.fixed_axes is None and params.relax_cell:
            params.fixed_axes = "None"
            self.audit.log_default(
                "fixed_axes",
                "None",
                "Relax all cell axes (no constraints)"
            )

        return params

    # ========== Relax-Specific Inference Rules ==========

    def _infer_relax_specific(
        self,
        params: RelaxParameters,
        context: Dict[str, Any]
    ) -> RelaxParameters:
        """
        Apply relax-specific inference rules.

        Infers:
        - relax_cell from presence of stress_thr or fixed_axes
        - Warnings for suboptimal parameter combinations
        """

        # Infer relax_cell if stress_thr or fixed_axes provided
        if params.relax_cell is False:
            if params.stress_thr is not None:
                self.audit.add_warning(
                    "stress_thr provided but relax_cell=False. "
                    "stress_thr is only used for cell-relax calculations."
                )
            if params.fixed_axes is not None and params.fixed_axes != "None":
                self.audit.add_warning(
                    f"fixed_axes={params.fixed_axes} provided but relax_cell=False. "
                    "fixed_axes is only used for cell-relax calculations."
                )

        # Warn if relax_cell=True but stress_thr not provided
        if params.relax_cell and params.stress_thr is None:
            self.audit.add_warning(
                "relax_cell=True but stress_thr not provided. "
                "Using default stress_thr=1.0 kBar."
            )

        return params


__all__ = [
    "RelaxDefaultsManager",
]
