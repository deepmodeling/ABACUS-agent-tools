"""Audit trail for DOS calculations."""
from typing import Optional
from ..common import BaseAuditLogger

class DOSAuditLogger(BaseAuditLogger):
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="dos", calculation_id=calculation_id)

__all__ = ["DOSAuditLogger"]
