"""Audit trail for Elastic calculations."""
from typing import Optional
from ..common import BaseAuditLogger
class ElasticAuditLogger(BaseAuditLogger):
    def __init__(self, calculation_id: Optional[str] = None):
        super().__init__(calculation_type="elastic", calculation_id=calculation_id)
__all__ = ["ElasticAuditLogger"]
