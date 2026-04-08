"""Parameter schemas for Elastic calculations."""
from typing import Optional
from dataclasses import dataclass
from ..common import CommonRelaxationParameters, SmearingMethod, MixingType
@dataclass
class ElasticParameters(CommonRelaxationParameters):
    """Schema for elastic calculation parameters."""
    norm_strain: Optional[float] = None
    shear_strain: Optional[float] = None
__all__ = ["ElasticParameters", "SmearingMethod", "MixingType"]
