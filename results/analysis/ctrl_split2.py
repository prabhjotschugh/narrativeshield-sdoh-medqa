import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from load import load_all, MODELS
sp=pd.read_csv("source_split.csv").set_index("question_id").split
D=load_all(); rows=[]; rng=np.random.default_rng(0)
for cond in ["B1","NS-full","NS-agent1only"]:
    for m in MODELS:
        x=D[(cond,m)].set_index("qid").join(sp)
        x["gap"]=x.alpha.astype(float)-(x.beta.astype(float)+x.gamma.astype(float))/2
        for s,g in x.groupby("split"):
            rows.append(dict(cond=cond,model=m,split=s,n=len(g),alpha=g.alpha.mean(),gap=g.gap.mean()))
R=pd.DataFrame(rows); R.to_csv("split_results.csv",index=False)
print(R.groupby(["cond","split"])[["n","alpha","gap"]].mean().round(3).to_string())
# pooled bootstrap CI for test-train difference in gap (resample questions, all models)
for cond in ["B1","NS-full","NS-agent1only"]:
    X=pd.concat([ (lambda x:x.assign(gap=x.alpha.astype(float)-(x.beta.astype(float)+x.gamma.astype(float))/2,model=m))(D[(cond,m)].set_index("qid").join(sp)) for m in MODELS])
    q=X.groupby(level=0).agg(gap=("gap","mean"),split=("split","first"))
    t=q[q.split=="test"].gap.values; r=q[q.split=="train"].gap.values
    diffs=[rng.choice(t,len(t)).mean()-rng.choice(r,len(r)).mean() for _ in range(5000)]
    print(f"{cond}: test-train gap diff {t.mean()-r.mean():+.3f}  95%CI [{np.percentile(diffs,2.5):+.3f},{np.percentile(diffs,97.5):+.3f}]  (test n={len(t)}, train n={len(r)})")
