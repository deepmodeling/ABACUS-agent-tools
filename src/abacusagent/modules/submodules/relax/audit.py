"""
Audit trail and provenance tracking for relax parameters.

This module implements the traceability principle:
- Every parameter value has documented origin
- Full audit trail from user input → defaults → inference → final value
- Human-readable summaries and machine-readable JSON output
- Inherits common audit functionality from the shared framework
"""

from typing import Optional
from ..common import BaseAuditLogger


class RelaxAuditLogger(BaseAuditLogger):
    """
    Tracks parameter provenance and creates audit trails for relax calculations.

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
        audit = RelaxAuditLogger()
        audit.log_user_input("force_thr_ev", 0.01, "Explicitly provided by user")
        audit.log_default("relax_nmax", 100, "Standard maximum relaxation steps")
        trail = audit.create_audit_trail()
        audit.print_summary()
    """

    def __init__(self, calculation_id: Optional[str] = None):
        """
        Initialize relax audit logger.

        Args:
            calculation_id: Optional unique ID for this calculation.
                           If not provided, generates a random 8-character ID.
        """
        super().__init__(calculation_type="relax", calculation_id=calculation_id)


__all__ = [
    "RelaxAuditLogger",
]
