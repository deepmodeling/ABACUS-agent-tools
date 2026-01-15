"""
Common parameter management framework.

This package provides the foundational components for parameter management
across all calculation modules. It includes:

- Base schemas for parameters and audit trails
- Common parameter definitions (enums and groups)
- Base validator with common validation methods
- Base audit logger for provenance tracking
- Base defaults manager with inference rules

Module-specific implementations inherit from these base classes and extend
them with module-specific logic.
"""

# Base schemas
from .base_schema import (
    BaseParameters,
    ParameterProvenance,
    ValidationResult,
    AuditTrail,
)

# Shared parameters
from .shared_parameters import (
    # Enums
    SmearingMethod,
    MixingType,
    BasisType,
    # Parameter groups
    ConvergenceParameters,
    SmearingParameters,
    MixingParameters,
    KPointParameters,
    ForceStressParameters,
    OutputParameters,
)

# Composable parameter groups
from .parameter_groups import (
    CommonSCFParameters,
    CommonRelaxationParameters,
    CommonPostSCFParameters,
)

# Base classes
from .base_validator import BaseParameterValidator
from .base_audit import BaseAuditLogger
from .base_defaults import BaseDefaultsManager, INFERENCE_RULES

__all__ = [
    # Base schemas
    "BaseParameters",
    "ParameterProvenance",
    "ValidationResult",
    "AuditTrail",
    # Enums
    "SmearingMethod",
    "MixingType",
    "BasisType",
    # Parameter groups
    "ConvergenceParameters",
    "SmearingParameters",
    "MixingParameters",
    "KPointParameters",
    "ForceStressParameters",
    "OutputParameters",
    # Composable groups
    "CommonSCFParameters",
    "CommonRelaxationParameters",
    "CommonPostSCFParameters",
    # Base classes
    "BaseParameterValidator",
    "BaseAuditLogger",
    "BaseDefaultsManager",
    # Inference rules
    "INFERENCE_RULES",
]
