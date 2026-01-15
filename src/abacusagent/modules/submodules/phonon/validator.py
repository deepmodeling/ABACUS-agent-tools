"""Validation logic for Phonon parameters."""
from typing import Dict, Tuple, List, Any
from ..common import BaseParameterValidator, ValidationResult
from .schema import PhononParameters
class PhononParameterValidator(BaseParameterValidator):
    def validate_all(self, params: PhononParameters, context: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        self.validation_results, self.warnings, self.errors = [], [], []
        self._validate_convergence_params(params)
        self._validate_smearing_params(params)
        self._validate_mixing_params(params)
        self._validate_kpoint_params(params)
        self._validate_output_params(params, context)
        return len(self.errors) == 0, self.validation_results
__all__ = ["PhononParameterValidator", "ValidationResult"]
