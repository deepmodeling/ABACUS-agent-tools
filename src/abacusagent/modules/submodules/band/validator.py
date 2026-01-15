"""Validation logic for band parameters."""
from typing import Dict, Tuple, List, Any
from ..common import BaseParameterValidator, ValidationResult
from .schema import BandParameters

class BandParameterValidator(BaseParameterValidator):
    """Validator for band parameters."""

    def validate_all(self, params: BandParameters, context: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        self.validation_results = []
        self.warnings = []
        self.errors = []

        self._validate_convergence_params(params)
        self._validate_smearing_params(params)
        self._validate_mixing_params(params)
        self._validate_kpoint_params(params)
        self._validate_output_params(params, context)
        self._validate_band_specific(params)

        return len(self.errors) == 0, self.validation_results

    def _validate_band_specific(self, params: BandParameters):
        if params.insert_point_nums is not None and params.insert_point_nums <= 0:
            self._add_error("insert_point_nums", f"insert_point_nums must be > 0, got {params.insert_point_nums}")

        if params.energy_min is not None and params.energy_max is not None:
            if params.energy_min >= params.energy_max:
                self._add_error("energy_min", f"energy_min must be < energy_max")

__all__ = ["BandParameterValidator", "ValidationResult"]
