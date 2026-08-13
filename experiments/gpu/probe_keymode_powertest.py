import json, sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
MAN=pd.read_parquet("manifests/aug_keymode_manifest.parquet")
def facts(f): return f if isinstance(f,dict) else json.loads(f)
def load(enc,task,layer):
    sub=MAN[MAN.task==task]; X,y,g=[],[],[]
    for r in sub.itertuples():
        p=Path(f"acts_keymode/{enc}")/(r.stimulus_id.replace("/","__")+".npz")
        if not p.exists(): continue
        z=np.load(p); k=f"pooled_{layer:02d}"
        if k not in z.files: return None,None,None
        X.append(z[k]); y.append(r.ground_truth); g.append(facts(r.factors)["soundfont"])
    return np.array(X),np.array(y),np.array(g)
def cv(X,y,g,mlp=False,yperm=None):
    yy=yperm if yperm is not None else y; accs=[]
    for tr,te in GroupKFold(3).split(X,yy,g):
        if len(np.unique(yy[tr]))<2: continue
        clf=make_pipeline(StandardScaler(), MLPClassifier((32,),early_stopping=True,max_iter=300) if mlp else LogisticRegression(max_iter=2000,C=1.0))
        clf.fit(X[tr],yy[tr]); accs.append((clf.predict(X[te])==yy[te]).mean())
    return np.mean(accs) if accs else np.nan
def run(enc,task):
    z0=np.load(next(Path(f"acts_keymode/{enc}").glob("*.npz")))
    nlayers=len([k for k in z0.files if k.startswith("pooled_")])
    ch=1/MAN[MAN.task==task].ground_truth.nunique()
    best=(-1,None,None)
    for L in range(nlayers):
        X,y,g=load(enc,task,L)
        if X is None: break
        a=max(cv(X,y,g,False),cv(X,y,g,True))
        if a>best[0]: best=(a,L,(X,y,g))
    a,L,(X,y,g)=best
    # permutation null on best layer (linear, faster)
    rng=np.random.default_rng(0)
    null=[cv(X,y,g,False,yperm=rng.permutation(y)) for _ in range(50)]
    null=np.array(null); p=(1+np.sum(null>=a))/(len(null)+1)
    print(f"  {enc} {task}: n={len(y)} best_acc={a:.3f} @L{L} (chance {ch:.3f}, +{a-ch:+.3f}); perm-null {np.nanmean(null):.3f}±{np.nanstd(null):.3f}, p={p:.3f}")
for task in ["mode_id","key_id"]:
    run(sys.argv[1], task)
