import numpy as np, pandas as pd
from load import *
D = load_all(); rng = np.random.default_rng(2)
CH = [m for m in MODELS if m != "openbiollm_8b"]
NB = 2000
def arr(c, m): return D[(c, m)][P].values.astype(float)
IDX = rng.integers(0, 1000, size=(NB, 1000))      # same resample for all comparisons
def gap_boot(A):   # (NB,) alpha - mean(beta,gamma)
    s = A[IDX].mean(1); return s[:,0] - (s[:,1]+s[:,2])/2
def acc_boot(A):
    return A.mean(1)[IDX].mean(1)

print("=== Gap reduction vs B1 (paired bootstrap, 2000 resamples, 11 models excl. OpenBioLLM) ===")
for c in ["B2","B3","B4","NS-full","NS-notool","NS-agent1only"]:
    red = []; sig = 0
    for m in CH:
        d = gap_boot(arr("B1", m)) - gap_boot(arr(c, m))
        red.append(d.mean()); sig += np.quantile(d, 0.025) > 0
    print("%-14s mean reduction %.3f   significant in %d/11 models" % (c, np.mean(red), sig))

print("\n=== Overall accuracy change vs B1 (points); * = 95% CI excludes 0 ===")
rows = []
for m in MODELS:
    r = {"model": m, "size": SIZE[m]}
    for c in ["B2","B3","B4","NS-full","NS-notool","NS-agent1only"]:
        d = acc_boot(arr(c, m)) - acc_boot(arr("B1", m)); lo, hi = np.quantile(d, [0.025, 0.975])
        r[c] = "%+.3f%s" % (d.mean(), "*" if (lo>0 or hi<0) else "")
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== NS variants head-to-head (11 models): accuracy change and gap ===")
for a, b in [("NS-full","NS-notool"), ("NS-full","NS-agent1only"), ("NS-notool","NS-agent1only")]:
    da, ns = [], 0
    for m in CH:
        d = acc_boot(arr(b, m)) - acc_boot(arr(a, m)); da.append(d.mean()); ns += (np.quantile(d,0.025)>0 or np.quantile(d,0.975)<0)
    print("%-10s -> %-14s mean accuracy change %+.3f (significant in %d/11)" % (a, b, np.mean(da), ns))

print("\n=== Size trend: NS-full accuracy change vs B1 ===")
ch = []
for m in CH:
    d = (acc_boot(arr("NS-full", m)) - acc_boot(arr("B1", m))).mean(); ch.append((SIZE[m], d))
ch = np.array(ch); print("corr(size, change) = %.2f" % np.corrcoef(ch[:,0], ch[:,1])[0,1])
for lo, hi in [(0,5),(5,9),(9,30)]:
    mk = (ch[:,0]>lo)&(ch[:,0]<=hi); print("  %2d-%2dB models: mean accuracy change %+.3f (n=%d)" % (lo, hi, ch[mk,1].mean(), mk.sum()))
print("\n=== CoT (B2) vs B1: accuracy change ===")
print("mean %+.3f, helps in %d/12" % (np.mean([arr("B2",m).mean()-arr("B1",m).mean() for m in MODELS]), sum(arr("B2",m).mean()>arr("B1",m).mean() for m in MODELS)))
