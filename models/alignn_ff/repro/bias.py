import gzip,csv,numpy as np,warnings; warnings.filterwarnings("ignore")
from matbench_discovery.data import df_wbm
FORM="e_form_per_atom_mp2020_corrected"; HULL="e_above_hull_mp2020_corrected_ppd_mp"
REF=df_wbm[[FORM,HULL]].dropna()
pred={}
with gzip.open("/home/kamalch/alignn2026/mbd_submission/2026-08-23-discovery.csv.gz","rt") as f:
    for i,r in enumerate(csv.reader(f)):
        if i: pred[r[0]]=float(r[1])
d=np.array([[pred[m],REF.loc[m,FORM],REF.loc[m,HULL]] for m in REF.index if m in pred])
efp,efd,eah=d[:,0],d[:,1],d[:,2]; k=np.abs(efp-efd)<5; efp,efd,eah=efp[k],efd[k],eah[k]
err=efp-efd
print(f"signed bias mean={err.mean():.4f} median={np.median(err):.4f} eV/atom  (negative = predicts too stable)")
# what if we subtract the median offset?
for off in [0, np.median(err)]:
    e2=efp-off; eahp=eah+(e2-efd); sp,sd=eahp<0,eah<0
    TP=int((sp&sd).sum());FP=int((sp&~sd).sum());FN=int((~sp&sd).sum())
    p=TP/(TP+FP);r=TP/(TP+FN);f1=2*p*r/(p+r); mae=np.abs(e2-efd).mean()
    print(f"  offset={off:+.4f}: F1={f1:.3f} Prec={p:.3f} Rec={r:.3f} MAE={mae:.3f}")
