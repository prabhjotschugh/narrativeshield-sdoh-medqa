import numpy as np, pandas as pd
from load import *
D = load_all()
rng = np.random.default_rng(1)

# 1) permutation null for NAG: shuffle persona labels within each question
def null_nag(df, n=1000):
    A = df[P].values.astype(float); out = np.empty(n)
    for i in range(n):
        S = A.copy()
        for r in range(len(S)):          # vectorised permutation per row
            pass
        idx = np.argsort(rng.random(S.shape), axis=1)
        S = np.take_along_axis(A, idx, axis=1)
        m = S.mean(0); out[i] = m[0] - min(m[1], m[2])
    return out.mean(), np.quantile(out, 0.95)
nn = []
for m in MODELS[:3]:
    nn.append(null_nag(D[("NS-full", m)])[0])
print("Null-NAG (no persona effect, n=1000 questions, 3 personas): mean %.4f  (95th pct %.4f)" % (np.mean(nn), null_nag(D[('NS-full','phi4_mini')])[1]))

# 2) decomposition of gap closure: B1 -> X.  d_alpha, d_marked(mean of beta,gamma)
conds = ["B2","B3","B4","NS-full","NS-notool","NS-agent1only"]
rows = []
for m in MODELS:
    b1 = D[("B1", m)]
    for c in conds:
        x = D[(c, m)]
        da = x.alpha.mean() - b1.alpha.mean()
        dm = np.mean([x.beta.mean()-b1.beta.mean(), x.gamma.mean()-b1.gamma.mean()])
        rows.append(dict(model=m, size=SIZE[m], cond=c, d_alpha=da, d_marked=dm,
                         gap_b1=b1.alpha.mean()-np.mean([b1.beta.mean(), b1.gamma.mean()]),
                         gap_x=x.alpha.mean()-np.mean([x.beta.mean(), x.gamma.mean()])))
R = pd.DataFrame(rows)
R["closure"] = R.gap_b1 - R.gap_x
R["share_from_alpha_drop"] = (-R.d_alpha) / R.closure
R.to_csv("decomp.csv", index=False)
S = R[R.model != "openbiollm_8b"]
print("\n=== mean change vs B1 (excluding OpenBioLLM), accuracy points ===")
print(S.groupby("cond")[["d_alpha","d_marked","gap_b1","gap_x","closure"]].mean().reindex(conds).round(3).to_string())
print("\n=== NS-full: per model change in control (alpha) vs marked (beta,gamma avg) ===")
f = R[R.cond=="NS-full"].sort_values("size")
print(f[["model","size","d_alpha","d_marked","gap_b1","gap_x"]].round(3).to_string(index=False))
print("\ncorr(model size, change in alpha under NS-full) = %.2f ; corr(size, change in marked) = %.2f" % (
    np.corrcoef(f["size"], f.d_alpha)[0,1], np.corrcoef(f["size"], f.d_marked)[0,1]))
for c in ["NS-notool","NS-agent1only"]:
    g = R[(R.cond==c)&(R.model!="openbiollm_8b")]
    print(c, "corr(size, d_alpha)=%.2f  corr(size, d_marked)=%.2f" % (np.corrcoef(g["size"], g.d_alpha)[0,1], np.corrcoef(g["size"], g.d_marked)[0,1]))
