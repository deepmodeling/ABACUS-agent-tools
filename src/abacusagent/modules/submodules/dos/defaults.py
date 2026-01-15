"""Default values for DOS parameters."""
from typing import Dict, Any
from copy import deepcopy
from ..common import BaseDefaultsManager
from .schema import DOSParameters
from .audit import DOSAuditLogger

class DOSDefaultsManager(BaseDefaultsManager):
    def __init__(self, audit_logger: DOSAuditLogger):
        super().__init__(audit_logger)

    def apply_defaults_and_inferences(self, params: DOSParameters, context: Dict[str, Any]) -> DOSParameters:
        params = deepcopy(params)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        return params

__all__ = ["DOSDefaultsManager"]
