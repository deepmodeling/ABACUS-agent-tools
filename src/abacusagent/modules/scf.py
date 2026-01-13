from pathlib import Path
from typing import Dict, Any, Optional, Literal

from abacusagent.init_mcp import mcp
from abacusagent.modules.submodules.scf import abacus_calculation_scf as _abacus_calculation_scf


@mcp.tool()
def abacus_calculation_scf(
    abacus_inputs_dir: Path,
    # Convergence parameters
    ecutwfc: Optional[float] = None,
    scf_thr: Optional[float] = None,
    scf_nmax: Optional[int] = None,
    # Smearing parameters
    smearing_method: Optional[Literal["gaussian", "fd", "fixed", "mp", "mv", "cold"]] = None,
    smearing_sigma: Optional[float] = None,
    # Mixing parameters
    mixing_type: Optional[Literal["plain", "kerker", "pulay", "pulay-kerker", "broyden"]] = None,
    mixing_beta: Optional[float] = None,
    mixing_ndim: Optional[int] = None,
    mixing_gg0: Optional[float] = None,
    # K-point parameters
    kspacing: Optional[float] = None,
    gamma_only: Optional[bool] = None,
    # Other parameters
    symmetry: Optional[bool] = None,
    out_chg: Optional[int] = None,
    out_mul: Optional[bool] = None,
    chg_extrap: Optional[Literal["none", "atomic", "first-order", "second-order"]] = None,
    ks_solver: Optional[Literal["cg", "dav", "bpcg", "genelpa", "scalapack_gvx"]] = None,
    # Audit control
    save_audit_trail: bool = True,
    print_audit_summary: bool = False,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation with explicit parameter control.

    This function supports two modes:
    1. Legacy mode: Only abacus_inputs_dir provided → uses INPUT file as-is
    2. New mode: Additional parameters provided → applies parameter management with full audit trail

    All parameters are optional - missing values will be filled with defaults or inferred.
    Full audit trail tracks parameter provenance (user input → defaults → inference → final value).

    Args:
        abacus_inputs_dir: Path to directory containing ABACUS input files (INPUT, STRU, KPT, etc.)

        Convergence parameters:
            ecutwfc: Energy cutoff for wavefunctions (Ry). Range: >0. Typical: 50-150 Ry
            scf_thr: SCF convergence threshold. Range: >0. Default: 1e-6
            scf_nmax: Maximum SCF iterations. Range: >0. Default: 100

        Smearing parameters:
            smearing_method: Electronic occupation smearing method.
                Options: gaussian (default), fd, fixed, mp, mv, cold
            smearing_sigma: Smearing width (Ry). Range: >0. Default: 0.015 Ry (≈0.2 eV)

        Mixing parameters:
            mixing_type: Charge density mixing method.
                Options: plain, kerker, pulay (default), pulay-kerker, broyden
            mixing_beta: Mixing parameter. Range: (0,1]. Default: depends on mixing_type
            mixing_ndim: Mixing history size (for pulay/broyden). Range: >0. Default: 8
            mixing_gg0: Kerker screening parameter (for kerker-based). Range: ≥0. Default: 0.0

        K-point parameters:
            kspacing: K-point spacing for automatic mesh (2π/Bohr). Range: >0
            gamma_only: Use only Gamma point. Default: False

        Other parameters:
            symmetry: Use crystal symmetry. Default: True
            out_chg: Output charge density. Options: 0 (no), 1 (yes), -1 (auto). Default: 0
            out_mul: Output Mulliken analysis. Default: False
            chg_extrap: Charge extrapolation method. Options: none, atomic, first-order, second-order
            ks_solver: Kohn-Sham solver. Options: cg, dav, bpcg, genelpa, scalapack_gvx

        Audit control:
            save_audit_trail: Save audit trail to JSON file. Default: True
            print_audit_summary: Print audit summary to console. Default: False

    Returns:
        Dictionary containing:
            - scf_work_dir: Path to calculation directory
            - normal_end: Whether calculation completed normally
            - converge: Whether SCF converged
            - energy: Final SCF energy (eV)
            - total_time: Calculation time (s)
            - audit_trail: Parameter provenance information (if save_audit_trail=True)

    Examples:
        # Legacy mode - use INPUT file as-is
        >>> result = abacus_calculation_scf("/path/to/inputs")

        # Custom convergence criteria
        >>> result = abacus_calculation_scf(
        ...     "/path/to/inputs",
        ...     ecutwfc=120,
        ...     scf_thr=1e-8,
        ...     scf_nmax=200
        ... )

        # Metal calculation with Kerker mixing
        >>> result = abacus_calculation_scf(
        ...     "/path/to/inputs",
        ...     smearing_method="mp",
        ...     smearing_sigma=0.02,
        ...     mixing_type="pulay-kerker",
        ...     mixing_gg0=1.5
        ... )
    """
    return _abacus_calculation_scf(
        abacus_inputs_dir=abacus_inputs_dir,
        ecutwfc=ecutwfc,
        scf_thr=scf_thr,
        scf_nmax=scf_nmax,
        smearing_method=smearing_method,
        smearing_sigma=smearing_sigma,
        mixing_type=mixing_type,
        mixing_beta=mixing_beta,
        mixing_ndim=mixing_ndim,
        mixing_gg0=mixing_gg0,
        kspacing=kspacing,
        gamma_only=gamma_only,
        symmetry=symmetry,
        out_chg=out_chg,
        out_mul=out_mul,
        chg_extrap=chg_extrap,
        ks_solver=ks_solver,
        save_audit_trail=save_audit_trail,
        print_audit_summary=print_audit_summary,
    )
