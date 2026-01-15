"""Default values for EOS parameters."""
from typing import Dict, Any
from copy import deepcopy
from ..common import BaseDefaultsManager
from .schema import EOSParameters
from .audit import EOSAuditLogger
class EOSDefaultsManager(BaseDefaultsManager):
    def __init__(self, audit_logger: EOSAuditLogger):
        super().__init__(audit_logger)
    def apply_defaults_and_inferences(self, params: EOSParameters, context: Dict[str, Any]) -> EOSParameters:
        params = deepcopy(params)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        return params
__all__ = ["EOSDefaultsManager"]
