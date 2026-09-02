import json,gzip,warnings,numpy as np; warnings.filterwarnings("ignore")
from ase import Atoms
from alignn.ff.ff import AlignnAtomwiseCalculator
calc=AlignnAtomwiseCalculator(path="/home/kamalch/alignn2026/mbd_submission/aff_model",
                              model_filename="best_model.pt", device="cpu")
DIST=np.round(np.linspace(0.5,6.0,56),4)
ELEMENTS=["H","Li","B","C","N","O","F","Na","Mg","Al","Si","P","S","Cl","K","Ca",
          "Ti","Fe","Ni","Cu","Zn","Ga","Ge","Mo","Ag","Sn","W","Pt","Au"]
homo={}
for el in ELEMENTS:
    E,F=[],[]
    for d in DIST:
        a=Atoms(el*2,positions=[[0,0,0],[0,0,float(d)]],cell=[30,30,30],pbc=False); a.calc=calc
        try: E.append(float(a.get_potential_energy())); F.append(a.get_forces().tolist())
        except Exception: E.append(float("nan")); F.append([[0,0,0],[0,0,0]])
    homo[f"{el}{el}"]={"energies":E,"forces":F}
curves={"distances":DIST.tolist(),"homo-nuclear":homo}
with gzip.open(f"/home/kamalch/alignn2026/mbd_submission/models/alignn_ff/alignn-ff/2026-08-23-diatomics.json.gz","wt") as f:
    json.dump(curves,f)
print("DIATOMICS_CURVES_DONE")
