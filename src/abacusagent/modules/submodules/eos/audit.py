"""Audit trail for EOS calculations."""
from typing import Optional
from ..common import BaseAuditLogger
class EOSAuditLogger(BaseAuditLogger):
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="eos", calculation_id=calculation_id)
__all__ = ["EOSAuditLogger"]
