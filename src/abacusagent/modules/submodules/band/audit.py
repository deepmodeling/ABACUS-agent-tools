"""Audit trail for band calculations."""
from typing import Optional
from ..common import BaseAuditLogger

class BandAuditLogger(BaseAuditLogger):
    """Audit logger for band calculations."""
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="band", calculation_id=calculation_id)

__all__ = ["BandAuditLogger"]
