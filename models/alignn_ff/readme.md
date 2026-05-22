# ALIGNN-FF

[ALIGNN-FF](https://arxiv.org/abs/2209.05554) is a universal interatomic potential built on
the [ALIGNN](https://arxiv.org/abs/2106.01829) (Atomistic Line Graph Neural Network)
architecture. ALIGNN performs message passing on both the atom graph and its line graph,
which encodes bond angles, allowing it to capture three-body interactions.

This submission evaluates the **default ALIGNN-FF force field** bundled with the
[`alignn`](https://pypi.org/project/alignn) package. The checkpoint is loaded via
`alignn.ff.ff.default_path()` and exposed to ASE through `AlignnAtomwiseCalculator`, so no
manual checkpoint download is required — installing `alignn` is sufficient.

## Scripts

| File | Task | Description |
| --- | --- | --- |
| [`test_alignn_ff_discovery.py`](test_alignn_ff_discovery.py) | discovery / geo-opt | Relaxes the WBM initial structures with ASE FIRE (positions + cell) and records final structures, energies and trajectories. |
| [`join_alignn_ff_preds.py`](join_alignn_ff_preds.py) | discovery / geo-opt | Concatenates the relaxation outputs, applies the MP2020 energy corrections and computes formation energies, producing `*-wbm-IS2RE.csv.gz` and `*-wbm-geo-opt-FIRE.jsonl.gz`. |
| [`test_alignn_ff_kappa.py`](test_alignn_ff_kappa.py) | phonons | Predicts lattice thermal conductivity κ for the PhononDB PBE-103 set via phono3py. |

## Reproducing

Each script declares its dependencies inline (PEP 723), so it can be run directly with
[`uv`](https://docs.astral.sh/uv):

```sh
# 1. relax the WBM test set (set TASK_ID / TASK_COUNT to split across nodes)
uv run test_alignn_ff_discovery.py

# 2. post-process into discovery + geo-opt submission files
uv run join_alignn_ff_preds.py

# 3. predict thermal conductivity
uv run test_alignn_ff_kappa.py
```

## Notes

An earlier ALIGNN-FF submission ([PR #47](https://github.com/janosh/matbench-discovery/pull/47))
was aborted in 2023 due to training-data incompatibility and resource limitations while
attempting to fine-tune the `alignnff_wt10` checkpoint on MPtrj. This submission instead
evaluates the released default ALIGNN-FF force field directly without fine-tuning.
