"""Numbers requested by the pre-submission review. Output: final/review_fixes_output.txt (printed)."""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
sys.path.insert(0, os.path.dirname(__file__))
from load import load_all, MODELS, ROOT

D = load_all()
FIN = os.path.join(os.path.dirname(__file__), "final")
mt = pd.read_csv(os.path.join(FIN, "master_table_final.csv"))
M11 = [m for m in MODELS if m != "openbiollm_8b"]
M8 = [m for m in M11 if ("NS-full", m) in D]
rng = np.random.default_rng(0)
CONDS = ["B1", "B2", "B3", "B4", "NS-agent1only", "NS-notool", "NS-full", "NS-storyform", "NS-rewrite"]

print("1. McNemar, alpha vs marked under B1: thresholds")
sec = pd.read_csv(os.path.join(ROOT, "b1_results", "b1_evaluation_secondary.csv")).set_index("Model")
ps = pd.concat([sec["McNemar p (a,b)"].astype(float), sec["McNemar p (a,g)"].astype(float)])
for thr, lab in [(0.05 / 3, "per model (3 pairs)"), (0.05 / 24, "across 24 tests"), (0.05 / 36, "across all 36 pairs")]:
    print(f"   {lab}: {int((ps < thr).sum())}/24 pass (thr {thr:.4f}); failing: {list(ps[ps >= thr].round(4).items())}")

print("\n2. Rewrite NAG vs null 95th pct (0.027) and CI")
r = mt[mt.cond == "NS-rewrite"]
print(f"   NAG > 0.027 in {int((r.nag > 0.027).sum())}/12; CI lower > 0 in {int((r.nag_lo > 0).sum())}/12; range {r.nag.min():.3f}-{r.nag.max():.3f}")
print(f"   11 models (no OpenBio): NAG>0.027 in {int((r[r.model!='openbiollm_8b'].nag > 0.027).sum())}/11, range {r[r.model!='openbiollm_8b'].nag.min():.3f}-{r[r.model!='openbiollm_8b'].nag.max():.3f}")
for c in ["NS-agent1only", "NS-storyform", "B4"]:
    s = mt[(mt.cond == c) & mt.model.isin(M11)]
    print(f"   {c}: NAG>0.027 in {int((s.nag > 0.027).sum())}/11")


def means(c, ms):
    s = mt[(mt.cond == c) & mt.model.isin(ms)]
    return s.a.mean(), ((s.b + s.g) / 2).mean(), s.overall.mean(), s.gap.mean()


print("\n3. Condition means on the same 8 models (those with +Tools), and on all 12 models")
rows = []
for c in CONDS:
    a8 = means(c, M8); a12 = means(c, MODELS) if c != "NS-full" else (np.nan,) * 4; a11 = means(c, M11)
    rows.append(dict(cond=c, a8=a8[0], mk8=a8[1], ov8=a8[2], gap8=a8[3], a11=a11[0], mk11=a11[1], ov11=a11[2], gap11=a11[3],
                     a12=a12[0], mk12=a12[1], ov12=a12[2], gap12=a12[3]))
R = pd.DataFrame(rows); print(R.round(3).to_string(index=False)); R.to_csv(os.path.join(FIN, "means_8_11_12.csv"), index=False)
print("   8-model set:", M8)

print("\n4. Deltas vs B1 (11 models, table means)")
b = R.set_index("cond").loc["B1"]
for c in CONDS[1:]:
    x = R.set_index("cond").loc[c]
    print(f"   {c:14s} d_alpha {100*(x.a11-b.a11):+.1f} d_marked {100*(x.mk11-b.mk11):+.1f} d_overall {100*(x.ov11-b.ov11):+.1f} gap {x.gap11:.3f} ({100*(1-x.gap11/b.gap11):.0f}% smaller)")
x = R.set_index("cond")
print(f"   Rewrite vs B4 overall {100*(x.loc['NS-rewrite'].ov11-x.loc['B4'].ov11):+.1f}; gap {100*(1-x.loc['NS-rewrite'].gap11/x.loc['B4'].gap11):.0f}% smaller")
print(f"   Rewrite vs Schema gap diff {x.loc['NS-rewrite'].gap11-x.loc['NS-agent1only'].gap11:+.3f}; overall {100*(x.loc['NS-rewrite'].ov11-x.loc['NS-agent1only'].ov11):+.1f}")

print("\n5. Schema: extraction parse rate and accuracy on parseable extractions only")
pr = []
for m in MODELS:
    f = os.path.join(ROOT, "ns_results_v2", f"results_NS_agent1only_{m}.jsonl")
    rw = D[("NS-rewrite", m)].set_index("qid")
    rec = []
    for l in open(f, encoding="utf-8"):
        o = json.loads(l)
        for p in ["alpha", "beta", "gamma"]:
            e = o["personas_eval"][p]
            rec.append(dict(qid=o["question_id"], p=p, parsed=bool(e.get("agent1_parse_success")), ok=bool(e["is_correct"]),
                            rw_ok=bool(rw.loc[o["question_id"], p])))
    X = pd.DataFrame(rec)
    allp = X.groupby("qid").parsed.all()
    Xp = X[X.qid.isin(allp[allp].index)]
    def gap(Z, col):
        g = Z.groupby("p")[col].mean(); return g["alpha"] - (g["beta"] + g["gamma"]) / 2
    pr.append(dict(model=m, parse=X.parsed.mean(), parse_a=X[X.p == "alpha"].parsed.mean(), parse_b=X[X.p == "beta"].parsed.mean(),
                   parse_g=X[X.p == "gamma"].parsed.mean(), n_q_allparsed=int(allp.sum()),
                   schema_acc_parsed=Xp.ok.mean(), rewrite_acc_same=Xp.rw_ok.mean(),
                   schema_gap_parsed=gap(Xp, "ok"), rewrite_gap_same=gap(Xp, "rw_ok")))
P = pd.DataFrame(pr); P.to_csv(os.path.join(FIN, "schema_parse.csv"), index=False)
print(P.round(3).to_string(index=False))
p11 = P[P.model != "openbiollm_8b"]
print(f"   11-model mean parse {p11.parse.mean():.3f} (alpha {p11.parse_a.mean():.3f}, beta {p11.parse_b.mean():.3f}, gamma {p11.parse_g.mean():.3f})")
print(f"   on questions where all three extractions parse: schema acc {p11.schema_acc_parsed.mean():.3f} vs rewrite {p11.rewrite_acc_same.mean():.3f} "
      f"(diff {100*(p11.rewrite_acc_same-p11.schema_acc_parsed).mean():+.1f}, rewrite better in {int((p11.rewrite_acc_same>p11.schema_acc_parsed).sum())}/11); "
      f"gap schema {p11.schema_gap_parsed.mean():.3f} rewrite {p11.rewrite_gap_same.mean():.3f}")

print("\n6. Rewrite recall vs correctness (audit sample)")
A = pd.read_csv(os.path.join(os.path.dirname(__file__), "rewrite_audit", "rewrite_audit.csv")).drop_duplicates(["model", "question_id", "persona"], keep="last")
A = A[(A.n_facts > 0) & (A.model != "openbiollm_8b")].copy()
A["recall"] = 1 - (A.n_missing / A.n_facts).clip(upper=1)
A["ok"] = [bool(D[("NS-rewrite", m)].set_index("qid").loc[q, p]) for m, q, p in zip(A.model, A.question_id, A.persona)]
A["full"] = A.recall >= 0.999
print("   recall by voice:", A.groupby("persona").recall.mean().round(3).to_dict())
print("   accuracy by recall bin:", A.groupby(pd.cut(A.recall, [0, .7, .85, .999, 1.0], include_lowest=True), observed=True).ok.agg(["mean", "size"]).round(3).to_dict())
rpb = pointbiserialr(A.ok, A.recall); print(f"   point-biserial r(recall, correct) = {rpb[0]:.3f} p={rpb[1]:.2g}")
import statsmodels.formula.api as smf
A["marked"] = (A.persona != "alpha").astype(int)
m1 = smf.ols("ok ~ marked + C(model) + C(question_id)", A.assign(ok=A.ok.astype(float))).fit()
m2 = smf.ols("ok ~ marked + recall + C(model) + C(question_id)", A.assign(ok=A.ok.astype(float))).fit()
print(f"   within-question marked effect on rewrite accuracy: {m1.params['marked']*100:+.1f} pts (p={m1.pvalues['marked']:.2g}); "
      f"controlling recall: {m2.params['marked']*100:+.1f} (p={m2.pvalues['marked']:.2g}); recall coef {m2.params['recall']:.3f} (p={m2.pvalues['recall']:.2g})")
inv = A.n_invented_relevant
m3 = smf.ols("ok ~ marked + recall + inv + C(model) + C(question_id)", A.assign(ok=A.ok.astype(float), inv=inv)).fit()
print(f"   + invented relevant items: marked {m3.params['marked']*100:+.1f}, invented coef {m3.params['inv']*100:+.2f} per item (p={m3.pvalues['inv']:.2g})")

print("\n7. OpenBioLLM included: 12-model means for main conditions")
for c in ["B1", "B4", "NS-agent1only", "NS-storyform", "NS-rewrite"]:
    a = means(c, MODELS); print(f"   {c:14s} alpha {a[0]:.3f} marked {a[1]:.3f} overall {a[2]:.3f} gap {a[3]:.3f}")

print("\n8. Leaky reference: alpha accuracy vs B1/B2")
o = means("LEAK-agent1only", M11); print(f"   leaky E-t-R (answer call saw JSON + original vignette): alpha {o[0]:.3f} marked {o[1]:.3f} overall {o[2]:.3f} gap {o[3]:.3f}")
