"""Parameter schemas for Phonon calculations."""
from typing import Optional, List, Dict
from dataclasses import dataclass
from ..common import CommonSCFParameters, SmearingMethod, MixingType
@dataclass
class PhononParameters(CommonSCFParameters):
    """Schema for phonon calculation parameters."""
    supercell: Optional[List[int]] = None
    displacement_stepsize: Optional[float] = None
__all__ = ["PhononParameters", "SmearingMethod", "MixingType"]
