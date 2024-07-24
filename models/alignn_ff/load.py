import os
from importlib.metadata import version
from matbench_discovery.data import DataFiles, as_dict_handler, df_wbm
from matbench_discovery.enums import MbdKey, Task
from matbench_discovery.plots import wandb_scatter
from pymatgen.core import Structure
from pymatviz.enums import Key
from typing import Any, Literal, Dict
import numpy as np
from tqdm import tqdm
from matbench_discovery.slurm import slurm_submit
import pandas as pd
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import pmg_to_atoms, ase_to_atoms
from alignn.ff.ff import AlignnAtomwiseCalculator
from ase.constraints import ExpCellFilter
from ase.optimize.fire import FIRE
from concurrent.futures import ThreadPoolExecutor, as_completed
from jarvis.db.jsonutils import loadjson, dumpjson
from matbench_discovery.energy import get_e_form_per_atom, mp_elemental_ref_energies

model_path = "/wrk/knc6/AFFBench/aff307k_lmdb_param_low_rad_use_force_mult_mp/out111continue5/"
model_filename = "current_model.pt"
calc = AlignnAtomwiseCalculator(
    path=model_path,
    force_mult_natoms=True,
    force_multiplier=1,
    stress_wt=0.3,
    modl_filename=model_filename,
    # stress_wt=-4800,
)


def general_relaxer(atoms="", calculator="", fmax=0.05, steps=500):
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calculator
    ase_atoms = ExpCellFilter(ase_atoms)

    dyn = FIRE(ase_atoms)
    dyn.run(fmax=fmax, steps=steps)
    en = ase_atoms.atoms.get_potential_energy()
    return en, ase_to_atoms(ase_atoms.atoms)


task_type = Task.IS2RE
data_path = {
    Task.RS2RE: DataFiles.wbm_computed_structure_entries.path,
    Task.IS2RE: DataFiles.wbm_initial_structures.path,
}[task_type]
input_col = {Task.IS2RE: Key.init_struct, Task.RS2RE: Key.final_struct}[
    task_type
]
df_in = pd.read_json(data_path).set_index(Key.mat_id)  # [0:10]
structures = df_in[input_col].map(Structure.from_dict).to_dict()
print(len(structures))
relax_results: dict[str, dict[str, Any]] = {}
count=0
for material_id, ii in tqdm(df_in.iterrows(), total=len(structures)):
    try:
        pmg = Structure.from_dict(ii["initial_structure"])
        atoms = pmg_to_atoms(pmg)
        if material_id in relax_results:
            continue
        en, struct = general_relaxer(atoms=atoms, calculator=calc)
        entry_like = dict(composition=pmg.composition.formula, energy=en)
        e_form = get_e_form_per_atom(entry_like,mp_elemental_ref_energies)
        relax_results[material_id] = {
            "energy": en,
            "initial_struct": atoms.to_dict(),
            "structure": struct.to_dict(),
            "e_form": e_form,
        }
        print(material_id, en, e_form)
        # print(struct)
        # break
        # except:
        # pass
        count+=1
        if count%1000==0:
           dumpjson(data=relax_results, filename="alignn_ff_tmp.json")
    except:
        pass
dumpjson(data=relax_results, filename="alignn_ff.json")
