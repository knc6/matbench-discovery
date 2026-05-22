# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pymatgen>=2024.5.1",
#   "tqdm>=4.66",
#   "matbench-discovery>=1.3.1",
# ]
#
# [tool.uv.sources]
# matbench-discovery = { path = "../../", editable = true }
# ///

"""Post-process ALIGNN-FF WBM relaxation outputs into submission files.

Concatenates the per-chunk `.json.gz` files written by `test_alignn_ff_discovery.py`,
applies the MP2020 energy corrections (structure-dependent for oxides/sulfides) and
computes corrected formation energies per atom. Produces two submission files:

- `<date>-wbm-IS2RE.csv.gz`: discovery task (material_id + e_form_per_atom_alignn_ff)
- `<date>-wbm-geo-opt-FIRE.jsonl.gz`: geometry-optimization task (relaxed structures
  + trajectories)
"""

# %%
import os
from glob import glob

import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatviz.enums import Key
from tqdm import tqdm

from matbench_discovery import today
from matbench_discovery.data import as_dict_handler, df_wbm
from matbench_discovery.energy import calc_energy_from_e_refs, mp_elemental_ref_energies
from matbench_discovery.enums import DataFiles, MbdKey

__author__ = "Kamal Choudhary, Janosh Riebesell"
__date__ = "2026-05-20"

module_dir = os.path.dirname(__file__)
e_form_col = "e_form_per_atom_alignn_ff"
struct_col = "alignn_ff_structure"
energy_col = "alignn_ff_energy"


# %% concatenate per-chunk relaxation outputs
glob_pattern = f"{module_dir}/alignn-ff/*-wbm-IS2RE-FIRE/*.json.gz"
file_paths = sorted(glob(glob_pattern))
print(f"Found {len(file_paths):,} files for {glob_pattern=}")
if not file_paths:
    raise SystemExit("No relaxation outputs found, run test_alignn_ff_discovery.py first")

df_alignn = pd.concat(
    pd.read_json(path, lines=True).set_index(Key.mat_id) for path in tqdm(file_paths)
)


# %% hydrate WBM computed structure entries
df_wbm_cse = pd.read_json(
    DataFiles.wbm_computed_structure_entries.path, lines=True
).set_index(Key.mat_id)
df_wbm_cse[Key.computed_structure_entry] = [
    ComputedStructureEntry.from_dict(dct)
    for dct in tqdm(df_wbm_cse[Key.computed_structure_entry], desc="Hydrate CSEs")
]


# %% transfer ALIGNN-FF energies and relaxed structures into the WBM CSEs so the
# structure-dependent MP2020 corrections are applied to the relaxed geometries
for mat_id, row in tqdm(
    df_alignn.iterrows(), total=len(df_alignn), desc="ML energies to CSEs"
):
    cse: ComputedStructureEntry = df_wbm_cse.loc[mat_id, Key.computed_structure_entry]
    cse._energy = row[energy_col]  # noqa: SLF001  uncorrected energy
    cse._structure = Structure.from_dict(row[struct_col])  # noqa: SLF001
    df_alignn.loc[mat_id, Key.computed_structure_entry] = cse


# %% apply MP2020 energy corrections
processed = MaterialsProject2020Compatibility().process_entries(
    df_alignn[Key.computed_structure_entry], verbose=True, clean=True
)
if len(processed) != len(df_alignn):
    raise ValueError(f"not all entries processed: {len(processed)=} {len(df_alignn)=}")


# %% compute corrected formation energies per atom
df_alignn[Key.formula] = df_wbm[Key.formula]
e_form_preds: dict[str, float] = {}
for mat_id, row in tqdm(
    df_alignn.iterrows(), total=len(df_alignn), desc="Formation energies"
):
    e_form_preds[mat_id] = calc_energy_from_e_refs(
        row[Key.formula],
        ref_energies=mp_elemental_ref_energies,
        total_energy=row[Key.computed_structure_entry].energy,
    )
df_alignn[e_form_col] = e_form_preds


# %% sanity check + write submission files
bad_mask = (df_alignn[e_form_col] - df_wbm[MbdKey.e_form_dft]).abs() > 5
print(f"{bad_mask.sum()=} ({bad_mask.mean():.2%}) predictions off by >5 eV/atom")

csv_path = f"{module_dir}/{today}-wbm-IS2RE.csv.gz"
df_alignn[[e_form_col]].round(4).to_csv(csv_path)
print(f"Wrote discovery predictions to {csv_path}")

json_path = f"{module_dir}/{today}-wbm-geo-opt-FIRE.jsonl.gz"
geo_opt_cols = [struct_col, energy_col, "alignn_ff_trajectory"]
df_alignn[[col for col in geo_opt_cols if col in df_alignn]].reset_index().to_json(
    json_path, default_handler=as_dict_handler, orient="records", lines=True
)
print(f"Wrote geometry-optimization structures to {json_path}")
