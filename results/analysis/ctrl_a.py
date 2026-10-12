import sys, difflib, numpy as np, pandas as pd
sys.path.insert(0,".")
from load import load_all, MODELS
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "medqa_1000.csv")
d = pd.read_csv(DS).set_index("question_id")
def wc(s): return len(str(s).split())
def sim(a,b): return difflib.SequenceMatcher(None,str(a).split(),str(b).split(),autojunk=False).ratio()
f = pd.DataFrame({"len_a":d.persona_alpha.map(wc),"len_b":d.persona_beta.map(wc),"len_g":d.persona_gamma.map(wc),
                  "sim_b":[sim(a,b) for a,b in zip(d.persona_alpha,d.persona_beta)],
                  "sim_g":[sim(a,b) for a,b in zip(d.persona_alpha,d.persona_gamma)],
                  "len_q":d.question.map(wc)})
f["dl_b"]=(f.len_b-f.len_a)/f.len_a; f["dl_g"]=(f.len_g-f.len_a)/f.len_a
print(f.describe().loc[["mean","std","min","50%","max"]].round(3).to_string())
f.to_csv("persona_text_features.csv")
D = load_all()
res=[]
for cond in ["B1","NS-full","NS-agent1only"]:
    for m in MODELS:
        x = D[(cond,m)].set_index("qid").join(f)
        # per-question gap: alpha - mean(beta,gamma) as 0/1 difference
        x["gap"] = x.alpha.astype(float) - (x.beta.astype(float)+x.gamma.astype(float))/2
        x["dl"]=(x.dl_b+x.dl_g)/2; x["sim"]=(x.sim_b+x.sim_g)/2
        res.append((cond,m,x))
# pooled regression: gap ~ dl + sim (+ model FE) using OLS with cluster-robust SE by question
import statsmodels.formula.api as smf
for cond in ["B1","NS-full","NS-agent1only"]:
    X = pd.concat([x.assign(model=m).reset_index() for c,m,x in res if c==cond])
    X["dl_z"]=(X.dl-X.dl.mean())/X.dl.std(); X["sim_z"]=(X.sim-X.sim.mean())/X.sim.std()
    r = smf.ols("gap ~ dl_z + sim_z + C(model)",X).fit(cov_type="cluster",cov_kwds={"groups":X.qid})
    print(f"\n[{cond}] pooled gap ~ length-change(z)+edit-similarity(z)+model FE; intercept(mean gap)={r.params['Intercept']:.3f}")
    for k in ["dl_z","sim_z"]: print(f"   {k}: coef {r.params[k]:+.4f}  95%CI [{r.conf_int().loc[k,0]:+.4f},{r.conf_int().loc[k,1]:+.4f}] p={r.pvalues[k]:.3g}")
# gap within tertiles of length change and of similarity (B1 & NS), pooled over models
for var in ["dl","sim"]:
    print(f"\nGap by tertile of {var} (pooled over 12 models; mean gap)")
    rows=[]
    for cond in ["B1","NS-full","NS-agent1only"]:
        X = pd.concat([x for c,m,x in res if c==cond])
        X["t"]=pd.qcut(X[var],3,labels=["low","mid","high"])
        rows.append(X.groupby("t",observed=True).gap.mean().rename(cond))
    print(pd.concat(rows,axis=1).round(3).to_string())
# length-matched: only questions where |dl_b|,|dl_g| in lowest half
print("\nB1 gap on shortest-added-length half vs longest half (pooled):")
X = pd.concat([x for c,m,x in res if c=="B1"]); med=X.dl.median()
print(" low-dl:",round(X[X.dl<=med].gap.mean(),3)," high-dl:",round(X[X.dl>med].gap.mean(),3))
