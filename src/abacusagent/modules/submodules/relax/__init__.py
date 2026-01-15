"""
Relax parameter management package.

This package implements schema-first, logic-explicit, and traceable
parameter management for ABACUS relax calculations.

Components:
- schema: Parameter schemas and type definitions
- validator: Validation logic and dependency rules
- audit: Audit trail and provenance tracking
- defaults: Default values and inference rules
"""

from .schema import (
    RelaxParameters,
    SmearingMethod,
    MixingType,
)
from .audit import RelaxAuditLogger
from .validator import (
    RelaxParameterValidator,
    ValidationResult,
)
from .defaults import RelaxDefaultsManager

__all__ = [
    "RelaxParameters",
    "SmearingMethod",
    "MixingType",
    "RelaxAuditLogger",
    "RelaxParameterValidator",
    "ValidationResult",
    "RelaxDefaultsManager",
]
