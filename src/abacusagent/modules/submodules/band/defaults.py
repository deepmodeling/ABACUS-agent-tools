"""Default values for band parameters."""
from typing import Dict, Any
from copy import deepcopy
from ..common import BaseDefaultsManager
from .schema import BandParameters
from .audit import BandAuditLogger

class BandDefaultsManager(BaseDefaultsManager):
    """Defaults manager for band parameters."""

    def __init__(self, audit_logger: BandAuditLogger):
        super().__init__(audit_logger)

    def apply_defaults_and_inferences(self, params: BandParameters, context: Dict[str, Any]) -> BandParameters:
        params = deepcopy(params)
        params = self._apply_convergence_defaults(params)
        params = self._apply_smearing_defaults(params, context)
        params = self._apply_mixing_defaults(params, context)
        params = self._apply_kpoint_defaults(params, context)
        params = self._apply_output_defaults(params, context)
        params = self._apply_band_defaults(params)
        params = self._infer_mixing_beta(params)
        params = self._infer_ks_solver(params, context)
        return params

    def _apply_band_defaults(self, params: BandParameters) -> BandParameters:
        if params.mode is None:
            params.mode = "auto"
            self.audit.log_default("mode", "auto", "Auto-detect band calculation mode")
        if params.energy_min is None:
            params.energy_min = -10.0
            self.audit.log_default("energy_min", -10.0, "Standard lower energy bound")
        if params.energy_max is None:
            params.energy_max = 10.0
            self.audit.log_default("energy_max", 10.0, "Standard upper energy bound")
        if params.insert_point_nums is None:
            params.insert_point_nums = 30
            self.audit.log_default("insert_point_nums", 30, "Standard k-point density")
        return params

__all__ = ["BandDefaultsManager"]
