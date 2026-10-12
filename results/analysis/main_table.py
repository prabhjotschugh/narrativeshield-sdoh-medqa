import numpy as np, pandas as pd
from load import *
from statsmodels.stats.contingency_tables import mcnemar
D = load_all()
rng = np.random.default_rng(0)

def boot_nag(df, n=2000):
    A = df[P].values.astype(float); N = len(A)
    idx = rng.integers(0, N, size=(n, N))
    res = np.empty(n)
    for i in range(n):
        m = A[idx[i]].mean(0)
        res[i] = m[0] - min(m[1], m[2])
    return np.quantile(res, [0.025, 0.975])

def mc(a, b):
    t = [[(a&b).sum(), (a&~b).sum()], [(~a&b).sum(), (~a&~b).sum()]]
    return mcnemar(t, exact=False, correction=True).pvalue

rows = []
for (cond, m), df in sorted(D.items()):
    a, b, g = (df[p].values for p in P)
    nag = a.mean() - min(b.mean(), g.mean())
    lo, hi = boot_nag(df)
    rows.append(dict(cond=cond, model=m, n=len(df), a=a.mean(), b=b.mean(), g=g.mean(),
                     gap_b=a.mean()-b.mean(), gap_g=a.mean()-g.mean(), nag=nag, nag_lo=lo, nag_hi=hi,
                     overall=np.mean([a.mean(), b.mean(), g.mean()]),
                     p_ab=mc(a,b), p_ag=mc(a,g), p_bg=mc(b,g),
                     parse=np.mean([df[p+"_parsed"].mean() for p in P])))
T = pd.DataFrame(rows)
T.to_csv("master_table.csv", index=False)
print(T.shape)
conds = ["B1","B2","B3","B4","NS-full","NS-notool","NS-agent1only"]
# summary across 12 chat models (exclude base)
S = T[T.model.isin(MODELS)]
print("\n=== mean over 12 chat models ===")
agg = S.groupby("cond").agg(overall=("overall","mean"), alpha=("a","mean"), beta=("b","mean"), gamma=("g","mean"),
                           nag=("nag","mean"), nag_med=("nag","median"),
                           nsig_ab=("p_ab", lambda x: (x < 0.05/3).sum()),
                           nsig_ag=("p_ag", lambda x: (x < 0.05/3).sum())).reindex(conds)
print(agg.round(3).to_string())
print("\n=== NAG per model (alpha - min(beta,gamma)); * = CI excludes 0 ===")
piv = T[T.model.isin(MODELS)].pivot(index="model", columns="cond", values="nag").reindex(MODELS)[conds]
star = T[T.model.isin(MODELS)].assign(s=lambda d: np.where(d.nag_lo>0,"*","")).pivot(index="model", columns="cond", values="s").reindex(MODELS)[conds]
out = piv.round(3).astype(str) + star
print(out.to_string())
print("\n=== overall accuracy per model ===")
print(T[T.model.isin(MODELS)].pivot(index="model", columns="cond", values="overall").reindex(MODELS)[conds].round(3).to_string())
