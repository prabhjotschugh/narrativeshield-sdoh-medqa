import numpy as np, pandas as pd
from load import *
D = load_all()
CH = [m for m in MODELS if m != "openbiollm_8b"]

print("=== Question-level transitions, B1 -> NS-full, marked persona (beta,gamma pooled) ===")
print("victim = control correct AND marked wrong under B1")
rows = []
for m in CH:
    b = D[("B1", m)]; n = D[("NS-full", m)]
    assert (b.qid.values == n.qid.values).all()
    vic = rec = both = broke = bothwrong = fixed = 0
    for p in ["beta", "gamma"]:
        a1 = b["alpha"].values; m1 = b[p].values; m2 = n[p].values
        v = a1 & ~m1;  vic += v.sum(); rec += (v & m2).sum()
        bc = a1 & m1;  both += bc.sum(); broke += (bc & ~m2).sum()
        bw = ~a1 & ~m1; bothwrong += bw.sum(); fixed += (bw & m2).sum()
    rows.append(dict(model=m, size=SIZE[m], victims=vic, recovered_pct=100*rec/vic,
                     collateral_pct=100*broke/both, newly_fixed_pct=100*fixed/bothwrong))
T = pd.DataFrame(rows)
print(T.round(1).to_string(index=False))
print("mean recovery %.1f%% | mean collateral damage %.1f%% | mean newly-fixed %.1f%%" % (T.recovered_pct.mean(), T.collateral_pct.mean(), T.newly_fixed_pct.mean()))

# baseline for 'recovery': how often does a B1-victim question come out right under B1 re-asked on ANOTHER persona?  use B3 (re-run w/ instruction) as a rough run-to-run-noise reference
print("\nReference: same victim questions under B3 (prompt-only debiasing)")
rec3 = []
for m in CH:
    b = D[("B1", m)]; n = D[("B3", m)]; v = r = 0
    for p in ["beta","gamma"]:
        vv = b["alpha"].values & ~b[p].values; v += vv.sum(); r += (vv & n[p].values).sum()
    rec3.append(100*r/v)
print("mean recovery under B3: %.1f%%" % np.mean(rec3))

print("\n=== Base vs instruct (llama31_8b) ===")
for c in ["B2","B3","B4"]:
    for m in ["llama31_8b_base","llama31_8b"]:
        d = D[(c, m)]; a,b_,g = (d[p].mean() for p in P)
        print("%s %-16s %.3f/%.3f/%.3f  NAG %.3f  parsed %.3f" % (c, m, a,b_,g, a-min(b_,g), np.mean([d[p+'_parsed'].mean() for p in P])))

print("\n=== Which persona is hardest? mean accuracy per persona, B1, 12 models ===")
print(pd.DataFrame({p: [D[("B1", m)][p].mean() for m in MODELS] for p in P}, index=MODELS).mean().round(3).to_string())
print("gamma < beta in %d/12 models under B1" % sum(D[("B1", m)].gamma.mean() < D[("B1", m)].beta.mean() for m in MODELS))
