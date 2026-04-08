"""
Base schema definitions for parameter management framework.

This module provides the foundational data structures used across all calculation
modules for parameter schemas, provenance tracking, and audit trails.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
import datetime


@dataclass
class BaseParameters:
    """
    Base class for all parameter schemas.

    All module-specific parameter classes should inherit from this base class.
    This provides a common interface for parameter handling across all calculation types.
    """
    pass


@dataclass
class ParameterProvenance:
    """
    Tracks the origin and reasoning for a single parameter value.

    This provides full traceability for how each parameter value was determined,
    enabling reproducibility and debugging.

    Attributes:
        parameter_name: Name of the parameter (e.g., "ecutwfc", "mixing_beta")
        value: The actual value assigned to the parameter
        source: How the value was determined:
            - "user_input": Explicitly provided by user
            - "default": Standard default value applied
            - "inferred": Inferred from other parameters via rules
            - "dependency": Set due to dependency constraint
        reasoning: Human-readable explanation of why this value was chosen
        timestamp: ISO format timestamp of when this provenance was recorded
        depends_on: List of parameter names this value depends on (for inferred/dependency)
        inference_rule: Named identifier for the inference rule used (for inferred)
    """
    parameter_name: str
    value: Any
    source: Literal["user_input", "default", "inferred", "dependency"]
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    depends_on: Optional[List[str]] = None
    inference_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ValidationResult:
    """
    Result of a single validation check.

    Attributes:
        is_valid: Whether the validation passed (False for errors, True for warnings/info)
        parameter: Name of the parameter being validated
        message: Human-readable validation message
        severity: Severity level:
            - "error": Blocks execution, invalid parameter
            - "warning": Allows execution, but flags potential issue
            - "info": Informational message, no issue
    """
    is_valid: bool
    parameter: str
    message: str
    severity: Literal["error", "warning", "info"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AuditTrail:
    """
    Complete audit trail for a calculation.

    This captures all parameter decisions, validation results, and issues
    for a single calculation, providing full traceability and reproducibility.

    Attributes:
        calculation_id: Unique identifier for this calculation
        calculation_type: Type of calculation (e.g., "scf", "relax", "band")
        parameters: Dictionary mapping parameter names to their provenance
        validation_results: List of all validation checks performed
        warnings: List of warning messages
        errors: List of error messages
    """
    calculation_id: str
    calculation_type: str
    parameters: Dict[str, ParameterProvenance]
    validation_results: List[ValidationResult]
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "parameters": {
                name: prov.to_dict()
                for name, prov in self.parameters.items()
            },
            "validation_results": [
                result.to_dict() if hasattr(result, 'to_dict') else result
                for result in self.validation_results
            ],
            "warnings": self.warnings,
            "errors": self.errors,
        }
