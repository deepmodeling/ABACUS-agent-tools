"""Default values for Elastic parameters."""
from typing import Dict, Any
from copy import deepcopy
from ..common import BaseDefaultsManager
from .schema import ElasticParameters
from .audit import ElasticAuditLogger
class ElasticDefaultsManager(BaseDefaultsManager):
    def __init__(self, audit_logger: ElasticAuditLogger):
        super().__init__(audit_logger)
    def apply_defaults_and_inferences(self, params: ElasticParameters, context: Dict[str, Any]) -> ElasticParameters:
        params = deepcopy(params)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        return params
__all__ = ["ElasticDefaultsManager"]
