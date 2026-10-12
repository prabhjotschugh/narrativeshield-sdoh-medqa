import sys, numpy as np, pandas as pd
sys.path.insert(0,".")
from datasets import load_dataset
from load import load_all, MODELS
ds = load_dataset("GBaker/MedQA-USMLE-4-options")
print({k:len(v) for k,v in ds.items()})
src={}
for sp in ds:
    for r in ds[sp]: src.setdefault(r["question"].strip(),sp)
d = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "medqa_1000.csv"))
d["split"]=d.question.str.strip().map(src)
print(d.split.value_counts(dropna=False))
d[["question_id","split"]].to_csv("source_split.csv",index=False)
D=load_all(); sp=d.set_index("question_id").split
rows=[]
for cond in ["B1","NS-full","NS-agent1only"]:
    for m in MODELS:
        x=D[(cond,m)].set_index("qid").join(sp)
        for s,g in x.groupby("split"):
            rows.append(dict(cond=cond,model=m,split=s,n=len(g),alpha=g.alpha.mean(),beta=g.beta.mean(),gamma=g.gamma.mean(),
                             gap=g.alpha.mean()-(g.beta.mean()+g.gamma.mean())/2))
R=pd.DataFrame(rows); R.to_csv("split_results.csv",index=False)
print(R.groupby(["cond","split"])[["n","alpha","beta","gamma","gap"]].mean().round(3).to_string())
# per-model paired comparison of gap train vs test (bootstrap-free: just diff)
P=R.pivot_table(index=["cond","model"],columns="split",values="gap")
P["diff"]=P.get("test",np.nan)-P.get("train",np.nan)
print(P.groupby("cond")["diff"].agg(["mean","std","min","max"]).round(3))
