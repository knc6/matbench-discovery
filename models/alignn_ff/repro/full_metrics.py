import gzip,csv,numpy as np,warnings; warnings.filterwarnings("ignore")
from matbench_discovery.data import df_wbm
FORM="e_form_per_atom_mp2020_corrected"; HULL="e_above_hull_mp2020_corrected_ppd_mp"
REF=df_wbm[[FORM,HULL]].dropna()
pred={}
with gzip.open("/home/kamalch/alignn2026/mbd_submission/2026-08-23-discovery.csv.gz","rt") as f:
    for i,row in enumerate(csv.reader(f)):
        if i==0: continue
        pred[row[0]]=float(row[1])
efp=[];efd=[];eah=[]
for mid in REF.index:
    if mid in pred:
        efp.append(pred[mid]); efd.append(REF.loc[mid,FORM]); eah.append(REF.loc[mid,HULL])
efp=np.array(efp);efd=np.array(efd);eah=np.array(eah)
k=np.abs(efp-efd)<5  # sanity filter
efp,efd,eah=efp[k],efd[k],eah[k]
eahp=eah+(efp-efd); sp,sd=eahp<0,eah<0
TP=int((sp&sd).sum());FP=int((sp&~sd).sum());TN=int((~sp&~sd).sum());FN=int((~sp&sd).sum())
prec=TP/(TP+FP);rec=TP/(TP+FN);f1=2*prec*rec/(prec+rec);acc=(TP+TN)/len(sp)
prev=sd.mean();daf=prec/prev
mae=np.abs(efp-efd).mean();rmse=np.sqrt(((efp-efd)**2).mean())
r2=1-((efp-efd)**2).sum()/((efd-efd.mean())**2).sum()
print(f"F1={f1:.3f} DAF={daf:.3f} Prec={prec:.3f} Rec={rec:.3f} Acc={acc:.3f}")
print(f"TP={TP} FP={FP} TN={TN} FN={FN}")
print(f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} n={len(efp)} missing={len(REF)-len(efp)}")
