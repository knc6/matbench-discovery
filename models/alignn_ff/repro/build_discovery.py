import json,gzip,csv
STG="/home/kamalch/alignn2026/mbd_submission"
muZ={int(k):v for k,v in json.load(open("/home/kamalch/alignn2026/wbm_fullcoverage/wbm_chempots_full_mptrj.json")).items()}
rows=[]
for line in open(f"{STG}/wbm_mptrj_smooth_condensed.jsonl"):
    r=json.loads(line); comp={int(z):c for z,c in r["comp"].items()}
    if any(z not in muZ for z in comp): continue
    ef=(r["e"]-sum(muZ[z]*c for z,c in comp.items()))/r["n"]
    rows.append((r["mid"], ef))
with gzip.open(f"{STG}/2026-08-23-discovery.csv.gz","wt",newline="") as f:
    w=csv.writer(f); w.writerow(["material_id","e_form_per_atom"])
    for mid,ef in rows: w.writerow([mid, f"{ef:.6f}"])
print(f"discovery file: {len(rows)} materials")
print("sample:", rows[:3])
