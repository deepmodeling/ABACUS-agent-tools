"""Audit trail for Phonon calculations."""
from typing import Optional
from ..common import BaseAuditLogger
class PhononAuditLogger(BaseAuditLogger):
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="phonon", calculation_id=calculation_id)
__all__ = ["PhononAuditLogger"]
