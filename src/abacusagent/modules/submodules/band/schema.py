"""
Parameter schemas and type definitions for band calculations.

This module defines the schema-first approach for band parameters:
- Inherits common parameters from the shared framework
- Adds band-specific parameters
- Explicit type hints using Literal types
"""

from typing import Literal, Optional, List, Dict, Union
from dataclasses import dataclass
from ..common import CommonPostSCFParameters


@dataclass
class BandParameters(CommonPostSCFParameters):
    """
    Schema for band calculation parameters.

    Inherits common parameters from CommonPostSCFParameters:
    - Convergence, Smearing, Mixing, K-points, Output

    Adds band-specific parameters:
    - mode: Calculation mode (nscf, pyatb, auto)
    - kpath: High symmetry k-point path
    - high_symm_points: Coordinates of high symmetry points
    - energy_min: Lower energy bound for plot
    - energy_max: Upper energy bound for plot
    - insert_point_nums: Points between high symmetry points
    """

    mode: Optional[Literal["nscf", "pyatb", "auto"]] = None
    """Band calculation mode (default: auto)"""

    kpath: Optional[Union[List[str], List[List[str]]]] = None
    """High symmetry k-point path"""

    high_symm_points: Optional[Dict[str, List[float]]] = None
    """Coordinates of high symmetry points"""

    energy_min: Optional[float] = None
    """Lower energy bound (eV, default: -10)"""

    energy_max: Optional[float] = None
    """Upper energy bound (eV, default: 10)"""

    insert_point_nums: Optional[int] = None
    """Points between high symmetry points (default: 30)"""


from ..common import SmearingMethod, MixingType

__all__ = ["BandParameters", "SmearingMethod", "MixingType"]
