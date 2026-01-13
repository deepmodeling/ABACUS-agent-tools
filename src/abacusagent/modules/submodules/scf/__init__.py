"""
SCF parameter management package.

This package implements schema-first, logic-explicit, and traceable
parameter management for ABACUS SCF calculations.

Components:
- schema: Parameter schemas and type definitions
- validator: Validation logic and dependency rules
- audit: Audit trail and provenance tracking
- defaults: Default values and inference rules
"""

from .schema import (
    SCFParameters,
    ParameterProvenance,
    SCFAuditTrail,
    SmearingMethod,
    MixingType,
    BasisType,
)
from .audit import SCFAuditLogger
from .validator import SCFParameterValidator, ValidationResult
from .defaults import SCFDefaultsManager

__all__ = [
    "SCFParameters",
    "ParameterProvenance",
    "SCFAuditTrail",
    "SmearingMethod",
    "MixingType",
    "BasisType",
    "SCFAuditLogger",
    "SCFParameterValidator",
    "ValidationResult",
    "SCFDefaultsManager",
]
