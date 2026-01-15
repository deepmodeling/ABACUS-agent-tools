"""Parameter schemas for EOS calculations."""
from typing import Optional
from dataclasses import dataclass
from ..common import CommonRelaxationParameters, SmearingMethod, MixingType
@dataclass
class EOSParameters(CommonRelaxationParameters):
    """Schema for EOS calculation parameters."""
    volume_range: Optional[float] = None
    num_points: Optional[int] = None
__all__ = ["EOSParameters", "SmearingMethod", "MixingType"]
