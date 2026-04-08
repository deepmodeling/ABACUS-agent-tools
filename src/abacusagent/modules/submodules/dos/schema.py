"""Parameter schemas for DOS calculations."""
from typing import Literal, Optional
from dataclasses import dataclass
from ..common import CommonPostSCFParameters, SmearingMethod, MixingType

@dataclass
class DOSParameters(CommonPostSCFParameters):
    """Schema for DOS calculation parameters."""
    dos_edelta_ev: Optional[float] = None
    dos_sigma: Optional[float] = None
    dos_scale: Optional[float] = None
    dos_emin_ev: Optional[float] = None
    dos_emax_ev: Optional[float] = None
    dos_nche: Optional[int] = None

__all__ = ["DOSParameters", "SmearingMethod", "MixingType"]
