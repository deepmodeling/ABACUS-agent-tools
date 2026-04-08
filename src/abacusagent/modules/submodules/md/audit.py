"""Audit trail for MD calculations."""
from typing import Optional
from ..common import BaseAuditLogger
class MDAuditLogger(BaseAuditLogger):
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="md", calculation_id=calculation_id)
__all__ = ["MDAuditLogger"]
