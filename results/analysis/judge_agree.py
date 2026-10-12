import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
sys.path.insert(0,".")
from load import load_all, MODELS
def get(p):
    a=pd.read_csv(p); a=a[a.n_added>=0].drop_duplicates(["question_id","persona"],keep="last"); return a.set_index(["question_id","persona"])
J={"gemini-3.8-flash":get("added_facts_audit.csv")}
for d in glob.glob("judge_*"):
    try: J[d[6:]]=get(d+"/added_facts_audit.csv")
    except Exception: pass
base="gemini-3.8-flash"
print("judge, n_scored")
for k,v in J.items(): print(" ",k,len(v))
rows=[]
for k,v in J.items():
    if k==base: continue
    j=J[base].join(v,lsuffix="_a",rsuffix="_b",how="inner")
    for p in ["alpha","beta","gamma","all"]:
        s=j if p=="all" else j.xs(p,level="persona")
        ba,bb=(s.n_clinically_relevant_a>0),(s.n_clinically_relevant_b>0)
        rows.append(dict(judge=k,persona=p,n=len(s),agree_any=(ba==bb).mean(),kappa_any=cohen_kappa_score(ba,bb),
            spearman_count=spearmanr(s.n_clinically_relevant_a,s.n_clinically_relevant_b)[0],
            rate_base=ba.mean(),rate_judge=bb.mean(),mean_rel_base=s.n_clinically_relevant_a.mean(),mean_rel_judge=s.n_clinically_relevant_b.mean()))
R=pd.DataFrame(rows); R.round(3).to_csv("judge_agreement.csv",index=False); print(R.round(3).to_string(index=False))
# gap by subset under each judge
Dd=load_all(); rng=np.random.default_rng(0)
def gap(cond):
    X=[]
    for m in MODELS:
        x=Dd[(cond,m)].set_index("qid"); x["gap"]=x.alpha.astype(float)-(x.beta.astype(float)+x.gamma.astype(float))/2; X.append(x[["gap"]])
    return pd.concat(X).groupby(level=0).gap.mean()
G={c:gap(c) for c in ["B1","NS-full"]}
out=[]
for k,v in J.items():
    w=v.n_clinically_relevant.unstack("persona").dropna(); w["mx"]=w[["beta","gamma"]].max(axis=1); w["avg"]=(w.beta+w.gamma)/2
    for name,mask in [("all",w.mx>=0),("zero added (both)",w.mx==0),("<=1 added (both)",w.mx<=1),(">=2 added (either)",w.mx>=2)]:
        ids=w.index[mask]
        for c,g in G.items():
            q=g.loc[g.index.intersection(ids)].values; b=[rng.choice(q,len(q)).mean() for _ in range(2000)]
            out.append(dict(judge=k,subset=name,cond=c,n_q=len(q),gap=q.mean(),lo=np.percentile(b,2.5),hi=np.percentile(b,97.5)))
O=pd.DataFrame(out); O.round(3).to_csv("gap_by_subset_all_judges.csv",index=False)
print(O[O.cond=="B1"].round(3).to_string(index=False))
