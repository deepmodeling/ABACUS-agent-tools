"""
Calculating elastic constants using ABACUS.
"""
import os
import shutil
import time
from typing import Dict, List
from pathlib import Path

import numpy as np
import dpdata
from pymatgen.core.structure import Structure
from pymatgen.analysis.elasticity.elastic import Strain
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.analysis.elasticity.strain import DeformedStructureSet
from pymatgen.analysis.elasticity.stress import Stress

from abacustest.lib_prepare.abacus import AbacusStru
from abacusagent.init_mcp import mcp
from abacusagent.modules.abacus import abacus_modify_input, abacus_collect_data
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, generate_work_path

def prepare_deformed_stru(
    input_stru_dir: Path,
    norm_strain: float,
    shear_strain: float
):
    """
    Generate deformed structures
    """
    temp = dpdata.System(os.path.join(input_stru_dir, "STRU"), fmt="abacus/stru")
    dump_poscar_name = Path('STRU-to-POSCAR-' + time.strftime("%Y%m%d%H%M%S"))
    temp.to('vasp/poscar', dump_poscar_name)
    original_stru = Structure.from_file(dump_poscar_name)
    os.remove(dump_poscar_name)

    norm_strains = [-norm_strain, -0.5*norm_strain, +0.5*norm_strain, +norm_strain]
    shear_strains = [-shear_strain, -0.5*shear_strain, +0.5*shear_strain, +shear_strain]
    deformed_strus = DeformedStructureSet(original_stru,
                                         symmetry=False,
                                         norm_strains=norm_strains,
                                         shear_strains=shear_strains)
    
    return deformed_strus

def prepare_deformed_stru_inputs(
    deformed_strus: DeformedStructureSet,
    work_path: Path,
    input_stru_dir: Path,
):
    """
    Prepare ABACUS inputs directories from deformed structures and prepared inputs templates
    """
    abacusjob_dirs = []
    copy_files = []
    for item in os.listdir(input_stru_dir):
        if os.path.isfile(os.path.join(input_stru_dir, item)):
            if item.endswith("INPUT") or item.endswith("KPT") or item.endswith(".orb")\
                    or item.endswith(".upf") or item.endswith(".UPF"):
                copy_files.append(item)
    
    original_stru = AbacusStru.ReadStru(os.path.join(input_stru_dir, "STRU"))

    stru_counts = 1
    for deformed_stru in deformed_strus:
        abacusjob_dir = os.path.join(work_path, f'deformed-stru-{stru_counts:0>3d}')
        os.mkdir(abacusjob_dir)
        for item in copy_files:
            shutil.copy(os.path.join(input_stru_dir, item), abacusjob_dir)
        
        # Write deformed structure to ABACUS STRU format
        dump_poscar_name = os.path.join(abacusjob_dir, 'deformed-STRU-POSCAR-' + time.strftime("%Y%m%d%H%M%S"))
        deformed_stru.to(dump_poscar_name, fmt='vasp/poscar')

        deformed_stru_poscar = dpdata.System(dump_poscar_name, fmt='vasp/poscar')
        os.remove(dump_poscar_name)
        first_dump_stru_name = os.path.join(abacusjob_dir, 'deformed-STRU-unmodified')
        deformed_stru_poscar.to('abacus/stru', first_dump_stru_name)

        deformed_stru_abacus = AbacusStru.ReadStru(first_dump_stru_name)
        os.remove(first_dump_stru_name)
        deformed_stru_abacus.set_pp(original_stru.get_pp())
        deformed_stru_abacus.set_orb(original_stru.get_orb())
        deformed_stru_abacus.set_atommag(original_stru.get_atommag())
        deformed_stru_abacus.write(os.path.join(abacusjob_dir, "STRU"))

        abacusjob_dirs.append(Path(abacusjob_dir).absolute())
        stru_counts += 1

    return abacusjob_dirs

def collected_stress_to_pymatgen_stress(stress: List[float]):
    """
    Transform calculated stress (units in kBar) collected by abacustest
    to Pymatgen format (units in GPa)
    """
    return Stress(-0.1 * np.array([stress[0:3],
                                   stress[3:6],
                                   stress[6: ]])) # 1 kBar = 0.1 GPa

@mcp.tool()
def abacus_cal_elastic(
    abacus_inputs_path: Path,
    norm_strain: float = 0.01,
    shear_strain: float = 0.01,
) -> Dict[str, float]:
    """
    Calculate various elastic constants for a given structure using ABACUS. 
    Args:
        abacus_inputs_path (str): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        norm_strain (float): Normal strain to calculate elastic constants, default is 0.01.
        shear_strain (float): Shear strain to calculate elastic constants, default is 0.01.
    Returns:
        A dictionary containing the following keys:
        elastic_constants (np.array in (6,6) dimension): Calculated elastic constants in Voigt notation. Units in GPa.
        bulk_modulus (float): Calculated bulk modulus in GPa.
        shear_modulus (float): Calculated shear modulus in GPa.
        young_modulus (float): Calculated Young's modulus in GPa.
        poisson_ratio (float): Calculated Poisson's ratio.
    Raises:
        RuntimeError: If ABACUS calculation when calculating stress for input structure or deformed structures fails.
    """
    abacus_inputs_path = Path(abacus_inputs_path).absolute()
    work_path = Path(generate_work_path()).absolute()
    input_stru_dir = Path(os.path.join(work_path, "input_stru")).absolute()

    link_abacusjob(src=abacus_inputs_path,
                   dst=input_stru_dir,
                   copy_files=["INPUT", "STRU", "KPT"])
    
    modified_params = {'calculation': 'scf',
                       'cal_stress': 1}
    modified_input = abacus_modify_input(input_stru_dir,
                                         extra_input = modified_params)
    
    deformed_strus = prepare_deformed_stru(input_stru_dir, norm_strain, shear_strain)
    strain = [Strain.from_deformation(d) for d in deformed_strus.deformations]

    deformed_stru_job_dirs = prepare_deformed_stru_inputs(deformed_strus, 
                                                          work_path,
                                                          input_stru_dir)
    
    all_dirs = [input_stru_dir] + deformed_stru_job_dirs
    run_abacus(all_dirs)

    collected_metrics = ["normal_end", "converge", "stress"]

    input_stru_result = abacus_collect_data(input_stru_dir, collected_metrics)
    if input_stru_result['collected_metrics']['converge'] is True:
        input_stru_stress = collected_stress_to_pymatgen_stress(input_stru_result['collected_metrics']['stress'])
    else:
        raise RuntimeError("SCF calculation for input structure doesn't converge")
    
    deformed_stru_stresses = []
    for idx, deformed_stru_job_dir in enumerate(deformed_stru_job_dirs):
        deformed_stru_result = abacus_collect_data(deformed_stru_job_dir, collected_metrics)
        if deformed_stru_result['collected_metrics']['converge'] is True:
            deformed_stru_stress = collected_stress_to_pymatgen_stress(deformed_stru_result['collected_metrics']['stress'])
            deformed_stru_stresses.append(deformed_stru_stress)
        else:
            raise RuntimeError(f"SCF calculation for deformed structure {idx} doesn't converge")
    
    result = ElasticTensor.from_independent_strains(strain,
                                                    deformed_stru_stresses,
                                                    eq_stress=input_stru_stress,
                                                    vasp=False)
    
    elastic_tensor = result.voigt.tolist()
    bv, gv = result.k_voigt, result.g_voigt
    ev = 9 * bv * gv / (3 * bv + gv)
    uV = (3 * bv - 2 * gv) / (6 * bv + 2 * gv)
    
    return {
        "elastic_cal_dir": Path(work_path).absolute(),
        "elastic_tensor": elastic_tensor,
        "bulk_modulus": float(bv),
        "shear_modulus": float(gv),
        "young_modulus": float(ev),
        "poisson_ratio": float(uV)
    }
