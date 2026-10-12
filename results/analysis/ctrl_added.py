import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from load import load_all, MODELS
import statsmodels.formula.api as smf
a=pd.read_csv("added_facts_audit.csv"); a=a[a.n_added>=0].drop_duplicates(["question_id","persona"],keep="last")
a.to_csv("added_facts_audit.csv",index=False)
print(a.persona.value_counts().to_dict())
s=a.groupby("persona").agg(n=("question_id","size"),mean_added=("n_added","mean"),mean_relevant=("n_clinically_relevant","mean"),pct_any_relevant=("n_clinically_relevant",lambda x:(x>0).mean()),pct_ge2=("n_clinically_relevant",lambda x:(x>=2).mean()))
s.round(3).to_csv("added_facts_summary.csv"); print(s.round(3).to_string())
w=a.pivot(index="question_id",columns="persona",values="n_clinically_relevant")
w["rel_marked"]=(w.beta+w.gamma)/2; w["rel_max"]=w[["beta","gamma"]].max(axis=1); w["rel_alpha"]=w.alpha
D=load_all(); rng=np.random.default_rng(0)
def pooled(cond):
    X=[]
    for m in MODELS:
        x=D[(cond,m)].set_index("qid"); x["gap"]=x.alpha.astype(float)-(x.beta.astype(float)+x.gamma.astype(float))/2; x["model"]=m; X.append(x)
    return pd.concat(X).join(w[["rel_marked","rel_max","rel_alpha"]])
def boot(v,n=3000):
    q=v.groupby(level=0).mean(); b=[rng.choice(q.values,len(q)).mean() for _ in range(n)]
    return q.mean(),np.percentile(b,2.5),np.percentile(b,97.5),len(q)
subsets={"all":lambda X:X.index==X.index,
 "marked adds 0 relevant (both)":lambda X:X.rel_max==0,
 "marked adds <=1 relevant (both)":lambda X:X.rel_max<=1,
 "marked adds <=1 relevant on avg":lambda X:X.rel_marked<=1,
 "marked adds >=2 relevant (either)":lambda X:X.rel_max>=2}
rows=[]
for cond in ["B1","B2","B3","B4","NS-full","NS-notool","NS-agent1only"]:
    X=pooled(cond)
    for k,f in subsets.items():
        sub=X[f(X)]
        m,lo,hi,n=boot(sub.gap); rows.append(dict(cond=cond,subset=k,n_q=n,gap=m,lo=lo,hi=hi))
R=pd.DataFrame(rows); R.to_csv("gap_by_added_subset.csv",index=False)
print(R.pivot(index="subset",columns="cond",values="gap").round(3).to_string())
print(R[R.cond=="B1"][["subset","n_q","gap","lo","hi"]].round(3).to_string(index=False))
for cond in ["B1","NS-full"]:
    X=pooled(cond).reset_index(); 
    r=smf.ols("gap ~ rel_marked + rel_alpha + C(model)",X).fit(cov_type="cluster",cov_kwds={"groups":X.qid})
    print(f"[{cond}] gap ~ relevant-additions(marked avg): coef {r.params['rel_marked']:+.4f} CI [{r.conf_int().loc['rel_marked',0]:+.4f},{r.conf_int().loc['rel_marked',1]:+.4f}] p={r.pvalues['rel_marked']:.3g}; intercept {r.params['Intercept']:+.3f}")
