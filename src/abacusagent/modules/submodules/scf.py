import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, collect_metrics
from .scf import (
    SCFParameters,
    SCFAuditLogger,
    SCFParameterValidator,
    SCFDefaultsManager,
    SmearingMethod,
    MixingType,
)


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
    2. New mode: Additional parameters provided → applies parameter management

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

    Raises:
        RuntimeError: If input files are invalid or validation fails
        Exception: If calculation fails

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
    try:
        # Validate input directory
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        # Create work directory and link files
        work_path = Path(generate_work_path()).absolute()
        link_abacusjob(src=abacus_inputs_dir, dst=work_path, copy_files=['INPUT', 'STRU'])

        # Read existing INPUT file
        input_params = ReadInput(os.path.join(work_path, "INPUT"))

        # Check if any SCF parameters were provided (new mode vs legacy mode)
        scf_params_provided = any([
            ecutwfc is not None,
            scf_thr is not None,
            scf_nmax is not None,
            smearing_method is not None,
            smearing_sigma is not None,
            mixing_type is not None,
            mixing_beta is not None,
            mixing_ndim is not None,
            mixing_gg0 is not None,
            kspacing is not None,
            gamma_only is not None,
            symmetry is not None,
            out_chg is not None,
            out_mul is not None,
            chg_extrap is not None,
            ks_solver is not None,
        ])

        audit_trail_dict = None

        if scf_params_provided:
            # New mode: Apply parameter management
            audit_trail_dict = _apply_parameter_management(
                input_params=input_params,
                work_path=work_path,
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
        else:
            # Legacy mode: Just set calculation type
            input_params['calculation'] = 'scf'
            WriteInput(input_params, os.path.join(work_path, "INPUT"))

        # Run ABACUS calculation
        run_abacus(work_path)

        # Collect results
        return_dict = {'scf_work_dir': Path(work_path).absolute()}
        return_dict.update(collect_metrics(
            work_path,
            metrics_names=['normal_end', 'converge', 'energy', 'total_time']
        ))

        # Add audit trail to results if available
        if audit_trail_dict is not None:
            return_dict['audit_trail'] = audit_trail_dict

        return return_dict

    except Exception as e:
        return {"message": f"Performing SCF calculation failed: {e}"}


def _apply_parameter_management(
    input_params: Dict[str, Any],
    work_path: Path,
    ecutwfc: Optional[float],
    scf_thr: Optional[float],
    scf_nmax: Optional[int],
    smearing_method: Optional[str],
    smearing_sigma: Optional[float],
    mixing_type: Optional[str],
    mixing_beta: Optional[float],
    mixing_ndim: Optional[int],
    mixing_gg0: Optional[float],
    kspacing: Optional[float],
    gamma_only: Optional[bool],
    symmetry: Optional[bool],
    out_chg: Optional[int],
    out_mul: Optional[bool],
    chg_extrap: Optional[str],
    ks_solver: Optional[str],
    save_audit_trail: bool,
    print_audit_summary: bool,
) -> Optional[Dict[str, Any]]:
    """
    Apply parameter management: parse, validate, infer, and update INPUT file.

    Returns:
        Audit trail dictionary if save_audit_trail=True, else None
    """
    # Initialize audit logger
    audit = SCFAuditLogger()

    # Parse user inputs into SCFParameters
    params = SCFParameters(
        ecutwfc=ecutwfc,
        scf_thr=scf_thr,
        scf_nmax=scf_nmax,
        smearing_method=SmearingMethod(smearing_method) if smearing_method else None,
        smearing_sigma=smearing_sigma,
        mixing_type=MixingType(mixing_type) if mixing_type else None,
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
        nspin=None,  # Will be filled from context
    )

    # Log user-provided parameters
    for param_name in [
        'ecutwfc', 'scf_thr', 'scf_nmax', 'smearing_method', 'smearing_sigma',
        'mixing_type', 'mixing_beta', 'mixing_ndim', 'mixing_gg0',
        'kspacing', 'gamma_only', 'symmetry', 'out_chg', 'out_mul',
        'chg_extrap', 'ks_solver'
    ]:
        value = getattr(params, param_name)
        if value is not None:
            audit.log_user_input(param_name, value)

    # Extract context from existing INPUT file
    context = {
        'basis_type': input_params.get('basis_type', 'lcao'),
        'soc': input_params.get('soc', False),
        'nspin': input_params.get('nspin', 1),
    }

    # Apply defaults and inferences
    defaults_mgr = SCFDefaultsManager(audit)
    params = defaults_mgr.apply_defaults_and_inferences(params, context)

    # Validate parameters
    validator = SCFParameterValidator()
    is_valid, validation_results = validator.validate_all(params, context)

    # Add validation results to audit
    for result in validation_results:
        audit.add_validation_result(result.to_dict())
    audit.warnings.extend(validator.warnings)
    audit.errors.extend(validator.errors)

    # If validation failed, raise error
    if not is_valid:
        error_msg = "Parameter validation failed:\n" + "\n".join(audit.errors)
        raise RuntimeError(error_msg)

    # Update INPUT parameters with validated SCF parameters
    _update_input_params(input_params, params)

    # Set calculation type to SCF
    input_params['calculation'] = 'scf'

    # Write updated INPUT file
    WriteInput(input_params, os.path.join(work_path, "INPUT"))

    # Print audit summary if requested
    if print_audit_summary:
        audit.print_summary()

    # Save audit trail if requested
    if save_audit_trail:
        audit.save_audit_trail(work_path)
        return audit.get_summary_dict()

    return None


def _update_input_params(input_params: Dict[str, Any], scf_params: SCFParameters):
    """
    Update INPUT parameters dictionary with SCF parameters.

    Args:
        input_params: Existing INPUT parameters (modified in-place)
        scf_params: Validated SCF parameters
    """
    # Convergence parameters
    if scf_params.ecutwfc is not None:
        input_params['ecutwfc'] = scf_params.ecutwfc
    if scf_params.scf_thr is not None:
        input_params['scf_thr'] = scf_params.scf_thr
    if scf_params.scf_nmax is not None:
        input_params['scf_nmax'] = scf_params.scf_nmax

    # Smearing parameters
    if scf_params.smearing_method is not None:
        smearing_str = scf_params.smearing_method.value if isinstance(scf_params.smearing_method, SmearingMethod) else scf_params.smearing_method
        input_params['smearing_method'] = smearing_str
    if scf_params.smearing_sigma is not None:
        input_params['smearing_sigma'] = scf_params.smearing_sigma

    # Mixing parameters
    if scf_params.mixing_type is not None:
        mixing_str = scf_params.mixing_type.value if isinstance(scf_params.mixing_type, MixingType) else scf_params.mixing_type
        input_params['mixing_type'] = mixing_str
    if scf_params.mixing_beta is not None:
        input_params['mixing_beta'] = scf_params.mixing_beta
    if scf_params.mixing_ndim is not None:
        input_params['mixing_ndim'] = scf_params.mixing_ndim
    if scf_params.mixing_gg0 is not None:
        input_params['mixing_gg0'] = scf_params.mixing_gg0

    # K-point parameters
    if scf_params.kspacing is not None:
        input_params['kspacing'] = scf_params.kspacing
    if scf_params.gamma_only is not None:
        input_params['gamma_only'] = 1 if scf_params.gamma_only else 0

    # Other parameters
    if scf_params.symmetry is not None:
        input_params['symmetry'] = 1 if scf_params.symmetry else 0
    if scf_params.out_chg is not None:
        input_params['out_chg'] = scf_params.out_chg
    if scf_params.out_mul is not None:
        input_params['out_mul'] = 1 if scf_params.out_mul else 0
    if scf_params.chg_extrap is not None:
        input_params['chg_extrap'] = scf_params.chg_extrap
    if scf_params.ks_solver is not None:
        input_params['ks_solver'] = scf_params.ks_solver
