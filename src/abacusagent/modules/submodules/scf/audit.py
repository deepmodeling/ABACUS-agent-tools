"""
Audit trail and provenance tracking for SCF parameters.

This module implements the traceability principle:
- Every parameter value has documented origin
- Full audit trail from user input → defaults → inference → final value
- Human-readable summaries and machine-readable JSON output
- Inherits common audit functionality from the shared framework
"""

from typing import Optional
from ..common import BaseAuditLogger


class SCFAuditLogger(BaseAuditLogger):
    """
    Tracks parameter provenance and creates audit trails for SCF calculations.

    Inherits all common audit functionality from BaseAuditLogger:
    - log_user_input(): Track user-provided parameters
    - log_default(): Track default values
    - log_inferred(): Track inferred values with inference rules
    - log_dependency(): Track dependency-driven values
    - print_summary(): Human-readable console output
    - save_audit_trail(): JSON serialization
    - get_summary_dict(): Summary statistics

    Design principle: Every parameter value must have a documented origin.

    Usage:
        audit = SCFAuditLogger()
        audit.log_user_input("ecutwfc", 100, "Explicitly provided by user")
        audit.log_default("scf_thr", 1e-6, "Standard convergence threshold")
        trail = audit.create_audit_trail()
        audit.print_summary()
    """

    def __init__(self, calculation_id: Optional[str] = None):
        """
        Initialize SCF audit logger.

        Args:
            calculation_id: Optional unique ID for this calculation.
                           If not provided, generates a random 8-character ID.
        """
        super().__init__(calculation_type="scf", calculation_id=calculation_id)


# ============================================================================
# RE-EXPORT COMMON TYPES FOR BACKWARD COMPATIBILITY
# ============================================================================

# Re-export common types so existing code can still import from this module
from ..common import ParameterProvenance, AuditTrail as BaseAuditTrail

# For backward compatibility with existing code that imports SCFAuditTrail
# Create a wrapper that provides the old interface (without calculation_type)
class SCFAuditTrail(BaseAuditTrail):
    """
    SCF-specific audit trail (backward compatibility wrapper).

    This is a thin wrapper around AuditTrail that automatically sets
    calculation_type='scf' for backward compatibility with existing code.
    """
    def __init__(
        self,
        calculation_id: str,
        parameters: dict,
        validation_results: list,
        warnings: list,
        errors: list
    ):
        """Initialize SCF audit trail with calculation_type='scf'."""
        super().__init__(
            calculation_id=calculation_id,
            calculation_type="scf",
            parameters=parameters,
            validation_results=validation_results,
            warnings=warnings,
            errors=errors
        )

__all__ = [
    "SCFAuditLogger",
    "ParameterProvenance",
    "SCFAuditTrail",
    "AuditTrail",
]
