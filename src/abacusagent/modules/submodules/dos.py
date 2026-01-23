import os
import glob
from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_collectdata.collectdata import RESULT
from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_model.comm_dos import DOSData, PDOSData, l_map, orbital_names

from pathlib import Path
from typing import Dict, Any, List, Literal, Optional, Tuple

from abacusagent.modules.util.comm import (
    generate_work_path,
    link_abacusjob,
    run_abacus,
    has_chgfile,
)



def abacus_dos_run(
    abacus_inputs_dir: Path,
    pdos_mode: Literal[
        "atoms", "species", "species+shell", "species+orbital"
    ] = "species+shell",
    pdos_atom_indices: Optional[List[int]] = None,
    dos_edelta_ev: float = 0.01,
    dos_sigma: float = 0.07,
    dos_emin_ev: float = -10.0,
    dos_emax_ev: float = 10.0,
) -> Dict[str, Any]:
    """Run the DOS and PDOS calculation.

    This function will firstly run a SCF calculation with out_chg set to 1,
    then run a NSCF calculation with init_chg set to 'file' and out_dos set to 1 or 2.
    If the INPUT parameter "basis_type" is "PW", then out_dos will be set to 1, and only DOS will be calculated and plotted.
    If the INPUT parameter "basis_type" is "LCAO", then out_dos will be set to 2, and both DOS and PDOS will be calculated and plotted.

    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        pdos_mode: Mode of plotted PDOS file.
            - "atoms": PDOS of a list of atoms will be plotted.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - "species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
        pdos_atom_indices: A list of atom indices, only used if pdos_mode is "atoms".
        dos_edelta_ev: Step size in writing Density of States (DOS) in eV.
        dos_sigma: Width of the Gaussian factor when obtaining smeared Density of States (DOS) in eV.
        dos_emin_ev: Minimal range for Density of States (DOS) in eV. Default is -10.0.
        dos_emax_ev: Maximal range for Density of States (DOS) in eV. Default is 10.0.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - dos_fig_path: Path to the plotted DOS.
            - pdos_fig_path: Path to the plotted PDOS. Only for LCAO basis.
            - dos_data_path: Path to the data used in plotting DOS.
            - pdos_data_paths: Path to the data used in plotting PDOS. Only for LCAO basis.
            - scf_work_path: Path to the work directory of SCF calculation.
            - scf_normal_end: If the SCF calculation ended normally.
            - scf_steps: Number of steps of SCF iteration.
            - scf_converge: If the SCF calculation converged.
            - scf_energy: The calculated energy of SCF calculation.
            - nscf_work_path: Path to the work directory of NSCF calculation.
            - nscf_normal_end: If the SCF calculation ended normally.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        input_file = os.path.join(abacus_inputs_dir, "INPUT")
        input_params = ReadInput(input_file)
        nspin = input_params.get("nspin", 1)
        if nspin in [4]:
            raise ValueError(
                "Currently DOS calculation can only be plotted using for nspin=1 and nspin=2"
            )
        
        print("Performing SCF calculation...")
        metrics_scf = abacus_dos_run_scf(abacus_inputs_dir)

        print("Performing NSCF calculation...")
        metrics_nscf = abacus_dos_run_nscf(
            metrics_scf["scf_work_path"],
            dos_edelta_ev=dos_edelta_ev,
            dos_sigma=dos_sigma,
        )

        fig_paths, dos_pdos_data_paths = plot_write_dos_pdos(
            metrics_scf["scf_work_path"],
            metrics_nscf["nscf_work_path"],
            pdos_mode,
            pdos_atom_indices,
            dos_emin_ev,
            dos_emax_ev,
        )

        return_dict = {"dos_fig_path": fig_paths[0]}
        return_dict["dos_data_path"] = dos_pdos_data_paths[0]
        try:
            return_dict["pdos_fig_path"] = fig_paths[1]
            return_dict["pdos_data_path"] = dos_pdos_data_paths[1]
        except:
            pass  # Do nothing if PDOS file is not plotted

        return_dict.update(metrics_scf)
        return_dict.update(metrics_nscf)

        return return_dict
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"message": f"Calculating DOS and PDOS failed: {e}"}


def abacus_dos_run_scf(
    abacus_inputs_dir: Path, force_run: bool = False
) -> Dict[str, Any]:
    """
    Run the SCF calculation to generate the charge density file.
    If the charge file already exists, it will skip the SCF calculation.

    Args:
        abacus_inputs_dir: Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        force_run: If True, it will run the SCF calculation even if the charge file already exists.

    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """

    input_param = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
    # check if charge file has been generated
    if has_chgfile(abacus_inputs_dir) and not force_run:
        print("Charge file already exists, skipping SCF calculation.")
        work_path = abacus_inputs_dir
    else:
        work_path = generate_work_path()
        link_abacusjob(src=abacus_inputs_dir, dst=work_path, copy_files=["INPUT"])

        input_param = ReadInput(os.path.join(work_path, "INPUT"))
        input_param["calculation"] = "scf"
        input_param["out_chg"] = 1
        WriteInput(input_param, os.path.join(work_path, "INPUT"))

        run_abacus(work_path)

    rs = RESULT(path=work_path, fmt="abacus")

    return {
        "scf_work_path": Path(work_path).absolute(),
        "scf_normal_end": rs["normal_end"],
        "scf_steps": rs["scf_steps"],
        "scf_converge": rs["converge"],
        "scf_energy": rs["energy"],
    }


def abacus_dos_run_nscf(
    abacus_inputs_dir: Path, dos_edelta_ev: float = None, dos_sigma: float = None
) -> Dict[str, Any]:
    work_path = generate_work_path()
    link_abacusjob(
        src=abacus_inputs_dir,
        dst=work_path,
        copy_files=["INPUT", "KPT"]
        + glob.glob(os.path.join(abacus_inputs_dir, "OUT.*")),
        exclude=["*log", "*json"],
    )

    input_param = ReadInput(os.path.join(work_path, "INPUT"))
    input_param["calculation"] = "nscf"
    input_param["init_chg"] = "file"
    input_param["out_chg"] = -1
    if input_param.get("basis_type", "pw") == "lcao":
        input_param["out_dos"] = 2  # only for LCAO basis, and will output DOS and PDOS
    else:
        input_param["out_dos"] = 1

    for dos_param, value in {
        "dos_edelta_ev": dos_edelta_ev,
        "dos_sigma": dos_sigma,
    }.items():
        if value is not None:
            input_param[dos_param] = value

    WriteInput(input_param, os.path.join(work_path, "INPUT"))

    run_abacus(work_path)

    rs = RESULT(path=work_path, fmt="abacus")

    return {
        "nscf_work_path": Path(work_path).absolute(),
        "nscf_normal_end": rs["normal_end"],
    }

def plot_write_dos_pdos(
    scf_job_path: Path,
    nscf_job_path: Path,
    mode: Literal[
        "species", "species+shell", "species+orbital", "atoms"
    ] = "species+shell",
    pdos_atom_indices: Optional[List[int]] = None,
    dos_emin_ev: float = -10.0,
    dos_emax_ev: float = 5.0,
) -> Tuple[List[str], List[str]]:
    """
    Plot DOS, PDOS and write data used in plotting to files using SCF and NSCF job directories from abacus_dos_run.

    Args:
        scf_job_path (Path): Path to the SCF job directory of the DOS calculation
        nscf_job_path (Path): Path to the NSCF job directory of the DOS calculation
        mode: Mode for plotting PDOS and write PDOS data.
            - "atoms": PDOS of a list of atoms will be plotted.
            - "species": Total PDOS of any species will be plotted in a picture.
            - "species+shell": PDOS for any shell (s, p, d, f, g,...) of any species will be plotted. PDOS of a shell of a species willbe plotted in a subplot.
            - "species+orbital": Orbital-resolved PDOS will be plotted. PDOS of orbitals in the same shell of a species will be plotted in a subplot.
        pdos_atom_indices: A list of atom indices, only used if pdos_mode is "atoms".
        pdos_atom_indices (List[int], optional): List of atom indices for atom-specific PDOS. Only valid for 'atoms' mode.
        dos_emin_ev (float): Minimum energy for DOS and PDOS plots.
        dos_emax_ev (float): Maximum energy for DOS and PDOS plots.    """
    work_path = generate_work_path()
    
    input_param = ReadInput(os.path.join(nscf_job_path, "INPUT"))
    basis_type = input_param.get("basis_type", "pw")
    
    results = RESULT(fmt="abacus", path=scf_job_path)
    efermi = results['efermi']

    # Construct file paths
    dos_plot_file = os.path.join(work_path, "DOS.png")
    dos_data_file = os.path.join(work_path, "DOS.dat")

    dosdata = DOSData.ReadFromAbacusJob(str(nscf_job_path), efermi)
    dosdata.plot_dos(
        dos_emin_ev,
        dos_emax_ev,
        "Density of States",
        dos_plot_file,
    )
    dosdata.write_dos(dos_data_file)

    all_plot_files = [Path(dos_plot_file).absolute()]
    dos_pdos_data_files = [Path(dos_data_file).absolute()]

    print("DOS file plotted")

    # Plot PDOS using PDOSData class (only for LCAO basis)
    if basis_type != "pw":
        try:
            # Load PDOS data using PDOSData class
            pdos_data = PDOSData.ReadFromAbacusJob(str(nscf_job_path), efermi)
            pdos_plot_file = Path(os.path.join(work_path, "PDOS.png")).absolute()
            pdos_data_file = Path(os.path.join(work_path, "PDOS.dat")).absolute()

            # Plot PDOS based on mode
            if mode == "species":
                pdos_data.plot_species_pdos(dos_emin_ev, dos_emax_ev, pdos_plot_file)
                pdos_data.write_species_pdos(pdos_data_file)
            elif mode == "species+shell":
                pdos_data.plot_species_shell_pdos(dos_emin_ev, dos_emax_ev, pdos_plot_file)
                pdos_data.write_species_shell_pdos(pdos_data_file)
            elif mode == "species+orbital":
                pdos_data.plot_species_orbital_pdos(dos_emin_ev, dos_emax_ev, pdos_plot_file)
                pdos_data.write_species_orbital_pdos(pdos_data_file)
            elif mode == "atoms":
                if pdos_atom_indices is None or len(pdos_atom_indices) == 0:
                    raise ValueError(
                        "For 'atoms' mode, pdos_atom_indices must be provided"
                    )
                pdos_data.plot_atoms_pdos(pdos_atom_indices, dos_emin_ev, dos_emax_ev, pdos_plot_file)
                pdos_data.write_atoms_pdos(pdos_atom_indices, pdos_data_file)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            # Combine file paths into a single list
            all_plot_files.append(Path(pdos_plot_file).absolute())
            dos_pdos_data_files.append(Path(pdos_data_file).absolute())

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Failed to plot PDOS: {e}")
            print("Skipping PDOS plotting")
    else:
        print(
            f"Warning: PDOS calculation not supported for PW basis type, skipping PDOS plotting"
        )

    print("Plots generated:")
    for file in all_plot_files:
        print(f"- {file}")

    return all_plot_files, dos_pdos_data_files
