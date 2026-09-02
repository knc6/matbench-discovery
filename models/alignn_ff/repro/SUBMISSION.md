# ALIGNN-FF → Matbench Discovery submission (revive aborted entry)

Reviving `models/alignn_ff` (was `lifecycle: aborted`) with the refreshed MPtrj
ALIGNN-FF (mptrj_smooth, 0.55 M params). Discovery task only (the core metric).

**Result (full WBM, MP2020-corrected):** F1 0.404, DAF 1.61, MAE 0.226 eV/atom.
Bottom-tier but a working entry (better than the current aborted state).

## Staged files
- `models/alignn_ff/alignn-ff.yml`  (updated: lifecycle active, params 547097, metrics, FIRE+FrechetCellFilter)
- `models/alignn_ff/alignn-ff/2026-08-23-discovery.csv.gz`  (256,963 rows: material_id, e_form_per_atom; md5 615c10b6677ed247d82b31b66705a746)

## Steps to submit
1. **Figshare**: upload `2026-08-23-discovery.csv.gz` (to project 279395), copy the
   direct-download file URL, and replace `FIGSHARE_URL_TO_FILL` in the yaml
   (`metrics.discovery.pred_file.url`).
2. **Checkpoint URL** (optional but recommended): the predictions are from the
   mptrj_smooth retrain, not the 2023 `alignnff_wt10`. Upload that checkpoint to
   Figshare too and point `checkpoint_url` at it (else leave the 2023 one + the note).
3. **Fork & branch**:
   ```
   git clone --depth 1 https://github.com/<you>/matbench-discovery
   cd matbench-discovery && git checkout -b alignn-ff-mptrj-refresh
   cp -r /home/kamalch/alignn2026/mbd_submission/models/alignn_ff/* models/alignn_ff/
   ```
4. **Validate**: `uv run scripts/ingest_model.py alignn-ff`  (recomputes metrics from the file)
5. **Commit & PR**: push the branch, open a PR to `janosh/matbench-discovery:main`,
   keep "allow maintainer edits" checked.

## Notes
- e_form uses self-consistent per-model elemental references over all 256,963 WBM
  structures (full coverage, incl. Sr/Cs/Gd/Th/Np).
- geo-opt task not included (our relax runs didn't record converged/n_steps; relaxed
  structures exist on the cluster if we want to add it later).
