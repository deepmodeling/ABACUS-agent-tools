import os
import re
import glob
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
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
    collect_metrics,
)
from abacusagent.modules.util.chemical_elements import MAX_ANGULAR_MOMENTUM_OF_ELEMENTS


angular_momentum_map = ["s", "p", "d", "f", "g"]
color_map = {
    "s": "#FF5733",
    "p": "#33FF57",
    "d": "#3357FF",
    "f": "#F033FF",
    "g": "#33FFF0",
}

orbital_rep_map = {
    "s": "s",
    "px": r"$p_x$",
    "py": r"$p_y$",
    "pz": r"$p_z$",
    "dz^2": r"$d_{z^2}$",
    "dxz": r"$d_{xz}$",
    "dyz": r"$d_{yz}$",
    "dxy": r"$d_{xy}$",
    "dx^2-y^2": r"$d_{x^2-y^2}$",
    "fz^3": r"$f_{z^3}$",
    "fxz^2": r"$f_{xz^2}$",
    "fyz^2": r"$f_{yz^2}$",
    "fzx^2-zy^2": r"$f_{zx^2-zy^2}$",
    "fxyz": r"$f_{xyz}$",
    "fx^3-3*xy^2": r"$f_{x^3-3xy^2}$",
    "f3yx^2-y^3": r"$f_{3yx^2-y^3}$",
}


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
            return_dict["pdos_data_paths"] = dos_pdos_data_paths[1:]
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


def plot_write_pdos_species(
    pdos_data: PDOSData,
    output_dir: Path,
    dos_emin_ev: float,
    dos_emax_ev: float,
) -> Tuple[str, List[str]]:
    """Plot PDOS by species using PDOSData class."""
    os.makedirs(output_dir, exist_ok=True)

    # Get unique species
    species_set = set()
    for orbital in pdos_data.projected_dos:
        species_set.add(orbital["species"])

    # Sum PDOS for each species
    species_pdos_data = []
    species_labels = []
    for species in species_set:
        species_pdos_data.append(pdos_data.get_pdos_by_species(species))
        species_labels.append(species)
    
    # Use PDOSData.plot_pdos method
    pdos_pic_file = os.path.join(output_dir, "PDOS.png")
    PDOSData.plot_pdos(
        pdosdatas=[species_pdos_data],
        labels=[species_labels],
        titles=["Projected density of States of different species"],
        energy=pdos_data.energy,
        energy_min=dos_emin_ev,
        energy_max=dos_emax_ev,
        pdos_fig_name=pdos_pic_file,
    )

    # Write data file using PDOSData.write_pdos method
    pdos_data_file = os.path.join(output_dir, "PDOS.dat")
    PDOSData.write_pdos(
        pdosdatas=species_pdos_data,
        energy=pdos_data.energy,
        labels=species_labels,
        filename=str(pdos_data_file),
    )

    return pdos_pic_file, pdos_data_file


def plot_write_pdos_species_shell(
    pdos_data: PDOSData,
    output_dir: Path,
    dos_emin_ev: float,
    dos_emax_ev: float,
) -> Tuple[str, List[str]]:
    """Plot PDOS by species and shell using PDOSData class."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique species
    species_set = set()
    for orbital in pdos_data.projected_dos:
        species_set.add(orbital["species"])

    # Prepare data for PDOSData.plot_pdos
    all_pdos_datas, all_labels, all_titles = [], [], []

    for species in sorted(species_set):
        species_pdos_data = []
        species_labels = []
        # Get shells for this species
        species_shells = []
        for orbital in pdos_data.projected_dos:
            if orbital["species"] == species:
                l = orbital["l"]
                if l not in species_shells:
                    species_shells.append(l)
                    species_shell_pdos = pdos_data.get_pdos_by_species_shell(species, l)
                    species_pdos_data.append(species_shell_pdos)

                    shell_label = f"{species}-{l_map[l]}"
                    species_labels.append(shell_label)
        
        all_titles.append(f"PDOS of {species}")
        all_pdos_datas.append(species_pdos_data)
        all_labels.append(species_labels)

    # Use PDOSData.plot_pdos method
    pdos_pic_file = os.path.join(output_dir, "PDOS.png")
    PDOSData.plot_pdos(
        pdosdatas=all_pdos_datas,
        labels=all_labels,
        titles=all_titles,
        energy=pdos_data.energy,
        energy_min=dos_emin_ev,
        energy_max=dos_emax_ev,
        pdos_fig_name=pdos_pic_file,
    )

    all_pdos_data_flattened = [data for species_pdos_datas in all_pdos_datas for data in species_pdos_datas]
    all_labels_flattened = [label for species_labels in all_labels for label in species_labels]
    pdos_data_file = os.path.join(output_dir, "PDOS.dat")
    PDOSData.write_pdos(
        pdosdatas=all_pdos_data_flattened,
        energy=pdos_data.energy,
        labels=all_labels_flattened,
        filename=pdos_data_file,
    )

    return pdos_pic_file, pdos_data_file


def plot_write_pdos_species_orbital(
    pdos_data: PDOSData,
    output_dir: Path,
    dos_emin_ev: float,
    dos_emax_ev: float,
) -> Tuple[str, List[str]]:
    """Plot PDOS by species and orbital using PDOSData class."""
    os.makedirs(output_dir, exist_ok=True)

    # Get unique species
    species_shell_set = set()
    for orbital in pdos_data.projected_dos:
        species_shell_set.add((orbital["species"], orbital["l"]))

    # Prepare data for PDOSData.plot_pdos
    all_pdos_datas, all_labels, all_titles = [], [], []

    for (species, l) in sorted(species_shell_set):
        orbital_pdos_data = []
        species_orbital_labels = []
        # Get orbitals for this shell
        species_orbitals = []
        for orbital in pdos_data.projected_dos:
            if orbital["species"] == species and orbital["l"] == l:
                m = orbital["m"]
                if m not in species_orbitals:
                    species_orbitals.append(m)
                    species_orbital_pdos = pdos_data.get_pdos_by_species_orbital(species, l, m)
                    orbital_pdos_data.append(species_orbital_pdos)

                    orbital_label = f"{species}-{orbital_names[(l, m)]}"
                    species_orbital_labels.append(orbital_label)
        
        all_titles.append(f"PDOS of {species}-{l_map[l]}")
        all_pdos_datas.append(orbital_pdos_data)
        all_labels.append(species_orbital_labels)

    # Use PDOSData.plot_pdos method
    pdos_pic_file = os.path.join(output_dir, "PDOS.png")
    PDOSData.plot_pdos(
        pdosdatas=all_pdos_datas,
        labels=all_labels,
        titles=all_titles,
        energy=pdos_data.energy,
        energy_min=dos_emin_ev,
        energy_max=dos_emax_ev,
        pdos_fig_name=pdos_pic_file,
    )

    all_pdos_data_flattened = [data for species_pdos_datas in all_pdos_datas for data in species_pdos_datas]
    all_labels_flattened = [label.replace("$", "") for species_labels in all_labels for label in species_labels]
    pdos_data_file = os.path.join(output_dir, "PDOS.dat")
    PDOSData.write_pdos(
        pdosdatas=all_pdos_data_flattened,
        energy=pdos_data.energy,
        labels=all_labels_flattened,
        filename=pdos_data_file,
    )

    return pdos_pic_file, pdos_data_file


def plot_write_pdos_atoms(
    pdos_data: PDOSData,
    output_dir: Path,
    pdos_atom_indices: List[int],
    dos_emin_ev: float,
    dos_emax_ev: float,
) -> Tuple[str, List[str]]:
    """Plot PDOS for selected atoms using PDOSData class."""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for PDOSData.plot_pdos
    all_pdos_datas, all_labels, all_titles = [], [], []

    for atom_index in pdos_atom_indices:
        # Obtain all shells of the selected atom
        atom_shell_set = set()
        for orbital in pdos_data.projected_dos:
            if orbital["atom_index"] == atom_index:
                species = orbital["species"]
                atom_shell_set.add((orbital["l"]))

        for l in atom_shell_set:
            orbital_pdos_data = []
            species_orbital_labels = []
            # Get shells for this species
            species_orbitals = []
            for orbital in pdos_data.projected_dos:
                if orbital["species"] == species and orbital["l"] == l:
                    m = orbital["m"]
                    if m not in species_orbitals:
                        species_orbitals.append(m)
                        species_orbital_pdos = pdos_data.get_pdos_by_species_orbital(species, l, m)
                        orbital_pdos_data.append(species_orbital_pdos)

                        orbital_label = f"{species}{atom_index}-{orbital_names[(l, m)]}"
                        species_orbital_labels.append(orbital_label)

            all_titles.append(f"PDOS of {species}{atom_index}-{l_map[l]}")
            all_pdos_datas.append(orbital_pdos_data)
            all_labels.append(species_orbital_labels)

    # Use PDOSData.plot_pdos method
    pdos_pic_file = os.path.join(output_dir, "PDOS.png")
    PDOSData.plot_pdos(
        pdosdatas=all_pdos_datas,
        labels=all_labels,
        titles=all_titles,
        energy=pdos_data.energy,
        energy_min=dos_emin_ev,
        energy_max=dos_emax_ev,
        pdos_fig_name=pdos_pic_file,
    )

    all_pdos_data_flattened = [data for species_pdos_datas in all_pdos_datas for data in species_pdos_datas]
    all_labels_flattened = [label.replace("$", "") for species_labels in all_labels for label in species_labels]
    pdos_data_file = os.path.join(output_dir, "PDOS.dat")
    PDOSData.write_pdos(
        pdosdatas=all_pdos_data_flattened,
        energy=pdos_data.energy,
        labels=all_labels_flattened,
        filename=pdos_data_file,
    )

    return pdos_pic_file, pdos_data_file

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
    """Plot DOS and PDOS from the NSCF job path using PDOSData class.

    Args:
        scf_job_path (Path): Path to the SCF job directory of the DOS calculation
        nscf_job_path (Path): Path to the NSCF job directory of the DOS calculation
        mode (str): PDOS plotting mode ('species', 'species+shell', 'species+orbital', or 'atoms').
        pdos_atom_indices (List[int], optional): List of atom indices for atom-specific PDOS. Only valid for 'atoms' mode.
        dos_emin_ev (float): Minimum energy for DOS and PDOS plots.
        dos_emax_ev (float): Maximum energy for DOS and PDOS plots.

    Returns:
        Tuple[List[str], List[str]]: Tuple containing list of plot file paths and data file paths.
    """
    work_path = generate_work_path()
    
    input_param = ReadInput(os.path.join(nscf_job_path, "INPUT"))
    basis_type = input_param.get("basis_type", "pw")
    
    results = RESULT(fmt="abacus", path=scf_job_path)
    efermi = results['efermi']

    # Construct file paths
    dos_plot_file = os.path.join(work_path, "DOS.png")
    dos_data_file = os.path.join(work_path, "DOS.dat")

    dosdata = DOSData.ReadFromAbacusJob(str(nscf_job_path), efermi)
    DOSData.plot_dos(
        dosdata.dosdata,
        dosdata.energy,
        dos_emin_ev,
        dos_emax_ev,
        "Density of States",
        dos_plot_file,
    )
    DOSData.write_dos(dosdata.dosdata, dosdata.energy, dos_data_file)

    all_plot_files = [Path(dos_plot_file).absolute()]
    dos_pdos_data_files = [Path(dos_data_file).absolute()]

    print("DOS file plotted")

    # Plot PDOS using PDOSData class (only for LCAO basis)
    if basis_type != "pw":
        try:
            # Load PDOS data using PDOSData class
            pdos_data = PDOSData.ReadFromAbacusJob(str(nscf_job_path), efermi)

            # Plot PDOS based on mode
            if mode == "species":
                pdos_plot_file, pdos_data_file = plot_write_pdos_species(
                    pdos_data, work_path, dos_emin_ev, dos_emax_ev
                )
            elif mode == "species+shell":
                pdos_plot_file, pdos_data_file = plot_write_pdos_species_shell(
                    pdos_data, work_path, dos_emin_ev, dos_emax_ev
                )
            elif mode == "species+orbital":
                pdos_plot_file, pdos_data_file = plot_write_pdos_species_orbital(
                    pdos_data, work_path, dos_emin_ev, dos_emax_ev
                )
            elif mode == "atoms":
                if pdos_atom_indices is None or len(pdos_atom_indices) == 0:
                    raise ValueError(
                        "For 'atoms' mode, pdos_atom_indices must be provided"
                    )
                pdos_plot_file, pdos_data_file = plot_write_pdos_atoms(
                    pdos_data,
                    work_path,
                    pdos_atom_indices,
                    dos_emin_ev,
                    dos_emax_ev,
                )
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
