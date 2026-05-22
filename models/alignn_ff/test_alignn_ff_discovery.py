# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "alignn>=2024.5.27",
#   "ase>=3.22.1",
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "torch>=2.2",
#   "pymatgen>=2024.5.1",
#   "tqdm>=4.66",
#   "matbench-discovery>=1.3.1",
# ]
#
# [tool.uv.sources]
# matbench-discovery = { path = "../../", editable = true }
# ///

"""Relax the WBM test set with the ALIGNN-FF universal force field.

For each WBM initial structure this script runs an ASE FIRE relaxation (atom
positions + cell) using the default ALIGNN-FF checkpoint shipped with the
`alignn` package (loaded via `alignn.ff.ff.default_path()`). The final relaxed
structure, energy and the full relaxation trajectory are written to disk.

The output `.json.gz` chunks are post-processed by `join_alignn_ff_preds.py`,
which applies the MP2020 energy corrections and computes formation energies to
produce the discovery (`*-wbm-IS2RE.csv.gz`) and geometry-optimization
(`*-wbm-geo-opt-FIRE.jsonl.gz`) submission files.
"""

# %%
import os
from copy import deepcopy
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
import torch
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from ase.filters import ExpCellFilter,FrechetCellFilter
from ase.optimize import FIRE
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery import today
from matbench_discovery.data import as_dict_handler, ase_atoms_from_zip
from matbench_discovery.enums import DataFiles, Task

from matgl.ext.ase import M3GNetCalculator
import matgl
from matgl.ext.ase import M3GNetCalculator
from chgnet.model.dynamics import CHGNetCalculator

alignn_calc =  CHGNetCalculator()


__author__ = "Kamal Choudhary, Philipp Benner, Janosh Riebesell"
__date__ = "2026-05-20"


# %% relaxation hyperparameters
task_type = Task.IS2RE
module_dir = os.path.dirname(__file__)
ase_optimizer = "FIRE"
max_steps = 500
force_max = 0.05  # run until forces are smaller than this in eV/Å
record_traj = True  # record intermediate structures into a pymatgen Trajectory
device = "cuda" if torch.cuda.is_available() else "cpu"

# optional job-array chunking: set TASK_ID in 1..TASK_COUNT to process one chunk
task_id = int(os.getenv("TASK_ID", "0"))
task_count = int(os.getenv("TASK_COUNT", "1"))

model_name = "alignn-ff"
job_name = f"{model_name}/{today}-wbm-{task_type}-{ase_optimizer}"
out_dir = f"{module_dir}/{job_name}"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/{task_id or 'all'}.json.gz"
if os.path.isfile(out_path):
    raise SystemExit(f"{out_path=} already exists, exiting early")


# %% load WBM initial structures
data_path = DataFiles.wbm_initial_atoms.path
print(f"Reading WBM initial structures from {data_path}")
atoms_list = ase_atoms_from_zip(data_path)

if task_count > 1 and task_id != 0:
    atoms_list = np.array_split(atoms_list, task_count)[task_id - 1]
    print(f"Relaxing chunk {task_id}/{task_count} with {len(atoms_list):,} structures")


# %% load ALIGNN-FF as an ASE calculator
print(f"Loading default ALIGNN-FF checkpoint on {device=}")
alignn_calc = AlignnAtomwiseCalculator(
    path=default_path(), stress_wt=0.01,device=device, include_stress=True
)

run_params = {
    "data_path": data_path,
    "versions": {dep: version(dep) for dep in ("alignn", "numpy", "torch", "ase")},
    "model_name": model_name,
    "task_type": task_type,
    "n_structures": len(atoms_list),
    "max_steps": max_steps,
    "force_max": force_max,
    "ase_optimizer": ase_optimizer,
    "device": device,
    "cell_filter": "ExpCellFilter",
    #"cell_filter": "FrechetCellFilter",
}
print(f"{run_params=}")


# %% relax structures
relax_results: dict[str, dict[str, Any]] = {}

for atoms in tqdm(deepcopy(atoms_list), desc="Relaxing with ALIGNN-FF"):
    mat_id = atoms.info[Key.mat_id]
    if mat_id in relax_results:
        continue
    try:
        atoms.calc = alignn_calc
        coords, lattices, energies = [], [], []

        init_struct = AseAtomsAdaptor.get_structure(atoms)
        print(f"\n=== {mat_id}: initial structure ===")
        print(init_struct)

        if max_steps > 0:
            filtered_atoms = ExpCellFilter(atoms)
            #filtered_atoms = FrechetCellFilter(atoms)
            optimizer = FIRE(filtered_atoms, logfile="-")
            if record_traj:
                optimizer.attach(lambda: coords.append(atoms.get_positions()))  # noqa: B023
                optimizer.attach(lambda: lattices.append(atoms.get_cell()))  # noqa: B023
                optimizer.attach(  # noqa: B023
                    lambda: energies.append(atoms.get_potential_energy())  # noqa: B023
                )
            optimizer.run(fmax=force_max, steps=max_steps)

        energy = atoms.get_potential_energy()  # relaxed total energy
        relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
        print(f"\n=== {mat_id}: optimized structure (energy={energy:.4f} eV) ===")
        print(relaxed_struct)
        relax_results[mat_id] = {"structure": relaxed_struct, "energy": energy}

        if record_traj and coords and lattices and energies:
            relax_results[mat_id]["trajectory"] = Trajectory(
                species=atoms.get_chemical_symbols(),
                coords=coords,
                lattice=lattices,
                constant_lattice=False,
                frame_properties=[{"energy": e} for e in energies],
            )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to relax {mat_id}: {exc!r}")


# %% save results
df_out = pd.DataFrame(relax_results).T.add_prefix("alignn_ff_")
df_out.index.name = Key.mat_id
df_out.reset_index().to_json(
    out_path, default_handler=as_dict_handler, orient="records", lines=True
)
print(f"Wrote {len(df_out):,} relaxed structures to {out_path}")
