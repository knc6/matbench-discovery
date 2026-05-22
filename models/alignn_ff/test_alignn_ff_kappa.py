# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "alignn>=2024.5.27",
#   "ase>=3.22.1",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "torch>=2.2",
#   "phono3py>=3.3",
#   "tqdm>=4.66",
#   "matbench-discovery>=1.3.1",
# ]
#
# [tool.uv.sources]
# matbench-discovery = { path = "../../", editable = true }
# ///

"""Predict lattice thermal conductivity (κ) for the PhononDB PBE-103 set with ALIGNN-FF.

Each structure is relaxed with ALIGNN-FF, then displaced and re-evaluated to build
the 2nd/3rd-order force constants from which phono3py computes κ. Results are written
as a `kappa-103-*.json.gz` file for the Matbench Discovery phonons task.
"""

# %%
import json
import os
import warnings
from datetime import datetime
from importlib.metadata import version
from typing import Any, Literal

import ase.io
import pandas as pd
import torch
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery import today
from matbench_discovery.enums import DataFiles
from matbench_discovery.phonons import KappaCalcParams, calc_kappa_for_structure

__author__ = "Kamal Choudhary, Janosh Riebesell"
__date__ = "2026-05-20"

module_dir = os.path.dirname(__file__)

# relaxation parameters
ase_optimizer: Literal["FIRE", "LBFGS", "BFGS"] = "FIRE"
max_steps = 300
force_max = 1e-4  # run until forces are smaller than this in eV/Å

# symmetry parameters
symprec = 1e-5  # symmetry precision for relaxation and conductivity calcs
enforce_relax_symm = True  # enforce symmetry during relaxation if broken
conductivity_broken_symm = False  # calc κ even if symmetry group changed on relaxation
save_forces = True  # save force sets to file
temperatures: list[float] = [300]
displacement_distance = 0.01

model_name = "alignn-ff"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading default ALIGNN-FF checkpoint on {device=}")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
alignn_calc = AlignnAtomwiseCalculator(
    path=default_path(), device=device, include_stress=True
)

job_name = (
    f"kappa-103-{ase_optimizer}-dist={displacement_distance}-"
    f"fmax={force_max}-{symprec=}"
)
out_dir = f"{module_dir}/{model_name}/{today}-{job_name}"
os.makedirs(out_dir, exist_ok=True)

timestamp = f"{datetime.now().astimezone():%Y-%m-%d@%H-%M-%S}"
print(f"\nJob {job_name} with {model_name} started {timestamp}")

atoms_list = ase.io.read(DataFiles.phonondb_pbe_103_structures.path, index=":")

kappa_params: KappaCalcParams = {
    "ase_optimizer": ase_optimizer,
    "max_steps": max_steps,
    "force_max": force_max,
    "symprec": symprec,
    "enforce_relax_symm": enforce_relax_symm,
    "conductivity_broken_symm": conductivity_broken_symm,
    "temperatures": temperatures,
    "out_dir": out_dir,
    "displacement_distance": displacement_distance,
    "save_forces": save_forces,
}
run_params = dict(
    **kappa_params,
    model_name=model_name,
    n_structures=len(atoms_list),
    struct_data_path=DataFiles.phonondb_pbe_103_structures.path,
    versions={dep: version(dep) for dep in ("numpy", "torch", "alignn")},
)
with open(f"{out_dir}/run_params.json", mode="w") as file:
    json.dump(run_params, file, indent=4)


# %% compute κ for each structure, saving intermediate results
kappa_results: dict[str, dict[str, Any]] = {}
force_results: dict[str, dict[str, Any]] = {}

for idx, atoms in enumerate(
    tqdm(atoms_list, desc=f"Predicting kappa with {model_name}")
):
    mat_id, result_dict, force_dict = calc_kappa_for_structure(
        atoms=atoms,
        calculator=alignn_calc,
        **kappa_params,
        task_id=idx,
    )
    kappa_results[mat_id] = result_dict
    if force_dict is not None:
        force_results[mat_id] = force_dict

    df_kappa = pd.DataFrame(kappa_results).T
    df_kappa.index.name = Key.mat_id
    df_kappa.reset_index().to_json(f"{out_dir}/kappa.json.gz")

    if save_forces:
        df_force = pd.concat([df_kappa, pd.DataFrame(force_results).T], axis=1)
        df_force.index.name = Key.mat_id
        df_force.reset_index().to_json(f"{out_dir}/force-sets.json.gz")

print(f"\nResults saved to {out_dir!r}")
