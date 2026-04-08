"""Parameter schemas for MD calculations."""
from typing import Literal, Optional
from dataclasses import dataclass
from ..common import CommonSCFParameters, SmearingMethod, MixingType

@dataclass
class MDParameters(CommonSCFParameters):
    """Schema for MD calculation parameters."""
    md_type: Optional[Literal["nve", "nvt", "npt", "langevin", "msst"]] = None
    md_nstep: Optional[int] = None
    md_dt: Optional[float] = None
    md_tfirst: Optional[float] = None
    md_tlast: Optional[float] = None
    md_thermostat: Optional[Literal["nhc", "anderson", "berendsen", "rescaling", "rescale_v"]] = None

__all__ = ["MDParameters", "SmearingMethod", "MixingType"]
