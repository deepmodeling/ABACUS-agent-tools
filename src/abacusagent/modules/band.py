from pathlib import Path
from typing import Literal, Dict, List, Union, Optional

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.band import abacus_cal_band as _abacus_cal_band
from abacusagent.modules.submodules.band import abacus_cal_effective_mass as _abacus_cal_effective_mass

@mcp.tool()
def abacus_cal_band(abacus_inputs_dir: Path,
                    mode: Literal["nscf", "pyatb", "auto"] = "auto",
                    kpath: Union[List[str], List[List[str]]] = None,
                    high_symm_points: Dict[str, List[float]] = None,
                    energy_min: float = -10,
                    energy_max: float = 10,
                    insert_point_nums: int = 30
) -> Dict[str, float|str]:
    """
    Calculate band using ABACUS based on prepared directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
    PYATB or ABACUS NSCF calculation will be used according to parameters in INPUT.
    Args:
        abacus_inputs_dir (str): Absolute path to a directory containing the INPUT, STRU, KPT, and pseudopotential or orbital files.
        mode: Method used to plot band. Should be `auto`, `pyatb` or `nscf`. 
            - `nscf` means using `nscf` calculation in ABACUS to calculate and plot the band
            - `pyatb` means using PYATB to plot the band
            - `auto` means deciding use `nscf` or `pyatb` mode according to the `basis_type` in INPUT file and files included in `abacus_inputs_dir`.
                -- If charge files are in `abacus_input_dir`, `nscf` mode will be used.
                -- If matrix files are in `abacus_input_dir`, `pyatb` mode will be used.
                -- If no matrix file or charge file are in `abacus_input_dir`, will determine mode by `basis_type`. If `basis_type` is lcao, will use `pyatb` mode.
                    If `basis_type` is pw, will use `nscf` mode.
        kpath (Tuple[List[str], List[List[str]]]): 
                A list of name of high symmetry points in the band path. Non-continuous line of high symmetry points are stored as seperate lists.
                For example, ['G', 'M', 'K', 'G'] and [['G', 'X', 'P', 'N', 'M', 'S'], ['S_0', 'G', R']] are both acceptable inputs.
                Default is None. If None, will use automatically generated kpath.
                `kpath` must be used with `high_symm_points` to take effect.
        high_symm_points: A dictionary containing high symmetry points and their coordinates in the band path. All points in `kpath` should be included.
                For example, {'G': [0, 0, 0], 'M': [0.5, 0.0, 0.0], 'K': [0.33333333, 0.33333333, 0.0], 'G': [0, 0, 0]}.
                Default is None. If None, will use automatically generated high symmetry points.
        energy_min (float): Lower bound of $E - E_F$ in the plotted band.
        energy_max (float): Upper bound of $E - E_F$ in the plotted band.
        insert_point_nums (int): Number of points to insert between two high symmetry points. Default is 30.
    Returns:
        A dictionary containing band gap, path to the work directory for calculating band and path to the plotted band.
    Raises:
    """
    return _abacus_cal_band(abacus_inputs_dir, mode, kpath, high_symm_points, energy_min, energy_max, insert_point_nums)


@mcp.tool()
def abacus_cal_effective_mass(
    band_calc_dir: Path,
    calculation_points: Union[Literal["auto", "extrema"], List[Dict[str, Union[str, List[float], int]]]] = "auto",
    fitting_window: int = 5,
    directions: List[str] = ["kx", "ky", "kz"],
    band_indices: Optional[List[int]] = None,
    energy_range: Optional[List[float]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Union[List, Path, Dict, str]]:
    """
    Calculate effective mass from band structure using parabolic fitting.

    This function analyzes band structure data to compute effective masses at band extrema
    (VBM/CBM) or user-specified k-points. It uses parabolic fitting E(k) = E₀ + a(k-k₀)²
    to extract the band curvature d²E/dk², from which the effective mass is calculated
    using m* = ℏ²/(d²E/dk²).

    Args:
        band_calc_dir: Path to directory containing band calculation results. Must have
            BANDS_*.dat files from ABACUS NSCF or PYATB calculation.
        calculation_points: Where to calculate effective mass. Options:
            - "auto": Automatically detect VBM and CBM (default)
            - "extrema": Find all local extrema within energy_range
            - List of dicts: User-specified points, each dict contains:
                * "type": "kpoint" or "high_symmetry"
                * "coords": [kx, ky, kz] for "kpoint" type
                * "label": "G", "M", "K", etc. for "high_symmetry" type
                * "band_index": (optional) specific band index
        fitting_window: Number of k-points on each side of extremum for parabolic
            fitting. Larger values give smoother fits but may miss non-parabolic
            behavior. Default: 5 (total 11 points used).
        directions: Directions for effective mass calculation. Currently only "kpath"
            direction (along the band path) is implemented. Default: ["kx", "ky", "kz"].
        band_indices: Specific band indices to analyze. If None, analyzes all bands
            near Fermi level. Default: None.
        energy_range: [E_min, E_max] in eV relative to Fermi level for extrema
            detection. Only used when calculation_points="extrema". Default: [-2, 2].
        output_dir: Directory for output files (plots and JSON). If None, uses
            band_calc_dir. Default: None.

    Returns:
        Dict containing:
            - effective_mass_results: List of dicts with effective mass data for each point.
              Each dict contains:
                * point_info: Band index, k-point index, coordinates, energy, extrema type
                * effective_masses: m* values in units of electron mass (m_e) for each direction
                * fitting_data: k-distances, energies, and fitted values for plotting
            - effective_mass_json: Path to JSON file with detailed results
            - effective_mass_plots: List of paths to visualization plots (one per point + summary)
            - summary: Dict with statistics:
                * electron_effective_mass: {average, min, max, std} for CBM
                * hole_effective_mass: {average, min, max, std} for VBM
            - message: Success or error message

    Raises:
        RuntimeError: If band data files are not found or cannot be read
        ValueError: If calculation_points format is invalid

    Examples:
        # Calculate effective mass at VBM and CBM automatically
        result = abacus_cal_effective_mass("/path/to/band/calc")
        print(f"Electron m* = {result['summary']['electron_effective_mass']['average']:.3f} m_e")

        # Calculate at all extrema within ±3 eV of Fermi level
        result = abacus_cal_effective_mass(
            "/path/to/band/calc",
            calculation_points="extrema",
            energy_range=[-3.0, 3.0]
        )

        # Calculate at specific k-point
        result = abacus_cal_effective_mass(
            "/path/to/band/calc",
            calculation_points=[
                {"type": "kpoint", "coords": [0.0, 0.0, 0.0], "band_index": 10}
            ]
        )

        # Calculate at high symmetry point
        result = abacus_cal_effective_mass(
            "/path/to/band/calc",
            calculation_points=[
                {"type": "high_symmetry", "label": "G"}
            ]
        )

    Notes:
        - Effective mass is reported in units of electron mass (m_e = 9.109×10⁻³¹ kg)
        - Positive curvature → positive effective mass (electron-like)
        - Negative curvature → negative effective mass (hole-like)
        - Reported values are absolute values |m*|
        - R² < 0.90 indicates poor parabolic fit; consider denser k-mesh
        - For accurate results, use insert_point_nums ≥ 30 in band calculation
    """
    return _abacus_cal_effective_mass(
        band_calc_dir,
        calculation_points,
        fitting_window,
        directions,
        band_indices,
        energy_range,
        output_dir
    )

