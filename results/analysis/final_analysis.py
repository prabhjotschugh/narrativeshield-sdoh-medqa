"""Final analysis. Every number in the paper comes from this script.
Writes CSVs to analysis/final/ and prints the summaries."""
import os, re, json, glob, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, pearsonr
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from load import load_all, MODELS, SIZE, ROOT

OUT = os.path.join(os.path.dirname(__file__), "final"); os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
D = load_all(include_leaky=True)
M11 = [m for m in MODELS if m != "openbiollm_8b"]
CONDS = ["B1", "B2", "B3", "B4", "NS-agent1only", "NS-notool", "NS-full", "NS-storyform", "NS-rewrite"]
LABEL = {"B1": "B1 direct", "B2": "B2 CoT", "B3": "B3 debias", "B4": "B4 CoT+debias", "NS-agent1only": "Extract-then-Reason",
         "NS-notool": "+Reason agent", "NS-full": "+Tools (full)", "NS-storyform": "Story+Form", "NS-rewrite": "Rewrite-then-Reason",
         "LEAK-agent1only": "oracle (leaky) E-t-R"}
B = 2000
rng = np.random.default_rng(0)
DS = pd.read_csv(os.path.join(ROOT, "..", "dataset", "medqa_1000.csv")).set_index("question_id")


def arr(cond, m):
    x = D[(cond, m)].set_index("qid").sort_index()
    return x


def mcnemar(a, b):
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def kappa(x, y):
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    po = (x == y).mean()
    cats = set(x) | set(y)
    pe = sum((x == c).mean() * (y == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


# ---------------- A. per (condition, model) master table ----------------
IDX = {}
rows = []
for (c, m), _ in sorted(D.items()):
    x = arr(c, m)
    a, b, g = (x[p].values.astype(float) for p in ["alpha", "beta", "gamma"])
    n = len(a)
    idx = rng.integers(0, n, (B, n))
    ma, mb, mg = a[idx].mean(1), b[idx].mean(1), g[idx].mean(1)
    nag_bs = ma - np.minimum(mb, mg)
    rows.append(dict(cond=c, model=m, n=n, a=a.mean(), b=b.mean(), g=g.mean(), overall=(a.mean() + b.mean() + g.mean()) / 3,
                     gap=a.mean() - (b.mean() + g.mean()) / 2, nag=a.mean() - min(b.mean(), g.mean()),
                     nag_lo=np.percentile(nag_bs, 2.5), nag_hi=np.percentile(nag_bs, 97.5),
                     p_ab=mcnemar(x.alpha.values, x.beta.values), p_ag=mcnemar(x.alpha.values, x.gamma.values),
                     kappa_ab=kappa(x.alpha_ans, x.beta_ans), kappa_ag=kappa(x.alpha_ans, x.gamma_ans),
                     parse=np.mean([x[p + "_parsed"].mean() for p in ["alpha", "beta", "gamma"]])))
MT = pd.DataFrame(rows)
MT.to_csv(os.path.join(OUT, "master_table_final.csv"), index=False)


# ---------------- B. paired comparisons vs a reference condition ----------------
def paired(c, ref, m):
    x, r = arr(c, m), arr(ref, m)
    q = x.index.intersection(r.index)
    x, r = x.loc[q], r.loc[q]
    def comps(t):
        a, b, g = (t[p].values.astype(float) for p in ["alpha", "beta", "gamma"])
        return a, (b + g) / 2
    xa, xm = comps(x); ra, rm = comps(r)
    n = len(q); idx = rng.integers(0, n, (B, n))
    dgap = (ra - rm) - (xa - xm)                  # positive = c has smaller gap than ref
    dacc = (xa + 2 * xm) / 3 - (ra + 2 * rm) / 3  # positive = c more accurate overall
    dmk = xm - rm; dal = xa - ra
    out = {}
    for k, v in [("gap_red", dgap), ("acc", dacc), ("marked", dmk), ("alpha", dal)]:
        bs = v[idx].mean(1)
        out[k] = v.mean(); out[k + "_lo"] = np.percentile(bs, 2.5); out[k + "_hi"] = np.percentile(bs, 97.5)
    return out


def summary(c, models, ref="B1"):
    s = MT[(MT.cond == c) & MT.model.isin(models)]
    ms = list(s.model)
    out = dict(cond=c, label=LABEL.get(c, c), n_models=len(ms), a=s.a.mean(), b=s.b.mean(), g=s.g.mean(),
               marked=((s.b + s.g) / 2).mean(), overall=s.overall.mean(), gap=s.gap.mean(), nag=s.nag.mean(),
               n_nag_sig=int((s.nag_lo > 0).sum()), kappa=((s.kappa_ab + s.kappa_ag) / 2).mean(), parse=s.parse.mean())
    if c != ref:
        P = pd.DataFrame([paired(c, ref, m) for m in ms])
        out.update(gap_red=P.gap_red.mean(), n_gap_red_sig=int((P.gap_red_lo > 0).sum()),
                   d_acc=P.acc.mean(), n_acc_better=int((P.acc_lo > 0).sum()), n_acc_worse=int((P.acc_hi < 0).sum()),
                   d_alpha=P.alpha.mean(), d_marked=P.marked.mean(),
                   n_marked_better=int((P.marked_lo > 0).sum()), n_marked_worse=int((P.marked_hi < 0).sum()))
    # transitions relative to B1
    if c != "B1":
        rec, brk = [], []
        for m in ms:
            b1, x = arr("B1", m), arr(c, m)
            q = x.index.intersection(b1.index)
            b1, x = b1.loc[q], x.loc[q]
            for p in ["beta", "gamma"]:
                v = b1.alpha & ~b1[p]; rec.append(x[p][v].mean())
                brk.append((~x[p][b1[p]]).mean())
        out.update(recovered=np.mean(rec), broken=np.mean(brk))
    return out


def pooled_effect(c, models):
    rows = []
    for m in models:
        x = D[(c, m)]
        for p in ["alpha", "beta", "gamma"]:
            rows.append(pd.DataFrame({"qid": x.qid, "model": m, "marked": int(p != "alpha"), "y": x[p].astype(float)}))
    Z = pd.concat(rows)
    Z["y_dm"] = Z.y - Z.groupby(["qid", "model"]).y.transform("mean")
    Z["m_dm"] = Z.marked - Z.groupby(["qid", "model"]).marked.transform("mean")
    r = smf.ols("y_dm ~ m_dm - 1", Z).fit(cov_type="cluster", cov_kwds={"groups": Z.qid})
    lo, hi = r.conf_int().loc["m_dm"]
    return r.params["m_dm"] * 100, lo * 100, hi * 100, r.pvalues["m_dm"]


S = []
for c in CONDS:
    ms = [m for m in M11 if (c, m) in D]
    d = summary(c, ms)
    d["pooled"], d["pooled_lo"], d["pooled_hi"], d["pooled_p"] = pooled_effect(c, ms)
    S.append(d)
SUM = pd.DataFrame(S)
SUM.to_csv(os.path.join(OUT, "condition_summary.csv"), index=False)
print("=== Condition summary (OpenBioLLM excluded; full = 8 models with 1000 rows) ===")
cols = ["label", "n_models", "a", "marked", "overall", "gap", "nag", "n_nag_sig", "pooled", "pooled_lo", "pooled_hi",
        "gap_red", "n_gap_red_sig", "d_acc", "n_acc_better", "n_acc_worse", "d_alpha", "d_marked", "n_marked_better",
        "n_marked_worse", "kappa", "recovered", "broken", "parse"]
print(SUM[cols].round(3).to_string(index=False))

# B1 on all 12 models (measurement claim)
b12 = summary("B1", MODELS); pe = pooled_effect("B1", MODELS)
print(f"\nB1, 12 models: a {b12['a']:.3f} b {b12['b']:.3f} g {b12['g']:.3f} NAG {b12['nag']:.3f} sig {b12['n_nag_sig']}/12; pooled {pe[0]:.1f} [{pe[1]:.1f},{pe[2]:.1f}]")
s12 = MT[(MT.cond == "B1")]
print(f"  min p over alpha-vs-marked McNemar (12 models): max p = {max(s12.p_ab.max(), s12.p_ag.max()):.2g}; Bonferroni 0.0167")

# ---------------- C. head-to-head: Rewrite vs others ----------------
print("\n=== Rewrite-then-Reason vs each condition (per-model paired bootstrap, 11 models) ===")
H = []
for ref in ["B1", "B2", "B3", "B4", "NS-agent1only", "NS-notool", "NS-storyform"]:
    P = pd.DataFrame([paired("NS-rewrite", ref, m) for m in M11])
    H.append(dict(ref=LABEL[ref], gap_red=P.gap_red.mean(), gap_better=int((P.gap_red_lo > 0).sum()), gap_worse=int((P.gap_red_hi < 0).sum()),
                  d_acc=P.acc.mean(), acc_better=int((P.acc_lo > 0).sum()), acc_worse=int((P.acc_hi < 0).sum()),
                  d_marked=P.marked.mean(), marked_better=int((P.marked_lo > 0).sum()), marked_worse=int((P.marked_hi < 0).sum())))
H = pd.DataFrame(H); H.to_csv(os.path.join(OUT, "rewrite_head_to_head.csv"), index=False)
print(H.round(3).to_string(index=False))

# ---------------- D. per-model rewrite table ----------------
print("\n=== Per model: B1 vs Rewrite-then-Reason ===")
PM = []
for m in MODELS:
    b, r = MT[(MT.cond == "B1") & (MT.model == m)].iloc[0], MT[(MT.cond == "NS-rewrite") & (MT.model == m)].iloc[0]
    pr = paired("NS-rewrite", "B1", m)
    PM.append(dict(model=m, size=SIZE[m], b1_a=b.a, b1_marked=(b.b + b.g) / 2, b1_gap=b.gap, b1_nag=b.nag,
                   rw_a=r.a, rw_marked=(r.b + r.g) / 2, rw_gap=r.gap, rw_nag=r.nag, rw_nag_lo=r.nag_lo, rw_nag_hi=r.nag_hi,
                   gap_red=pr["gap_red"], gap_red_lo=pr["gap_red_lo"], d_acc=pr["acc"], d_acc_lo=pr["acc_lo"], d_acc_hi=pr["acc_hi"],
                   d_marked=pr["marked"], rw_parse=r.parse, b1_kappa=(b.kappa_ab + b.kappa_ag) / 2, rw_kappa=(r.kappa_ab + r.kappa_ag) / 2))
PM = pd.DataFrame(PM); PM.to_csv(os.path.join(OUT, "rewrite_per_model.csv"), index=False)
print(PM.round(3).to_string(index=False))
pm11 = PM[PM.model != "openbiollm_8b"]
print(f"corr(size, gap reduction) = {pearsonr(pm11['size'], pm11.gap_red)[0]:.2f}; corr(size, acc change) = {pearsonr(pm11['size'], pm11.d_acc)[0]:.2f}")
print(f"corr(B1 gap, gap reduction) = {pearsonr(pm11.b1_gap, pm11.gap_red)[0]:.2f}  (bigger bias -> bigger fix?)")
print(f"share of B1 gap removed, per model: mean {np.mean(pm11.gap_red / pm11.b1_gap):.2f}, range {np.min(pm11.gap_red / pm11.b1_gap):.2f} to {np.max(pm11.gap_red / pm11.b1_gap):.2f}")

# per voice
for c in ["NS-rewrite", "NS-storyform", "NS-agent1only", "B4"]:
    db = np.mean([arr(c, m).beta.mean() - arr("B1", m).beta.mean() for m in M11])
    dg = np.mean([arr(c, m).gamma.mean() - arr("B1", m).gamma.mean() for m in M11])
    print(f"{LABEL[c]:22s} change vs B1: beta {db*100:+.1f}  gamma {dg*100:+.1f} points")


# ---------------- E. robustness subsets ----------------
A = pd.read_csv("age_preserved.csv").set_index("question_id")
aud = pd.read_csv("added_facts_audit.csv").pivot(index="question_id", columns="persona", values="n_clinically_relevant")
sp = pd.read_csv("source_split.csv").set_index("question_id").split
L = pd.read_csv("verbatim_leak.csv").set_index("question_id")
F = pd.read_csv("persona_text_features.csv", index_col=0); dl = (F.dl_b + F.dl_g) / 2
le1 = aud[["beta", "gamma"]].max(axis=1) <= 1
SUBS = {"all": DS.index, "shorter half": dl.index[dl <= dl.median()], "test split": sp.index[sp == "test"],
        "age explicit both": A.index[A.both], "<=1 relevant addition": aud.index[le1],
        "strict clean": A.index[A.both & le1]}


def qgap(c, models, persona=None):
    cols = []
    for m in models:
        x = arr(c, m)
        mk = (x.beta.astype(float) + x.gamma.astype(float)) / 2 if persona is None else x[persona].astype(float)
        cols.append(x.alpha.astype(float) - mk)
    return pd.concat(cols, axis=1).mean(axis=1)


def bci(v):
    v = np.asarray(v); bs = v[rng.integers(0, len(v), (B, len(v)))].mean(1); return np.percentile(bs, [2.5, 97.5])


print("\n=== Gap on controlled subsets (11 models) ===")
R = []
for c in ["B1", "B4", "NS-agent1only", "NS-storyform", "NS-rewrite"]:
    G = qgap(c, M11)
    for k, ids in SUBS.items():
        v = G.loc[G.index.intersection(ids)].values; lo, hi = bci(v)
        R.append(dict(cond=LABEL[c], subset=k, n=len(v), gap=v.mean(), lo=lo, hi=hi))
    Gb = qgap(c, M11, "beta"); v = Gb.loc[L.index[L.beta]].values; lo, hi = bci(v)
    R.append(dict(cond=LABEL[c], subset="beta superset (alpha-beta)", n=len(v), gap=v.mean(), lo=lo, hi=hi))
R = pd.DataFrame(R); R.to_csv(os.path.join(OUT, "subset_gaps.csv"), index=False)
print(R.pivot(index="subset", columns="cond", values="gap").round(3).to_string())


# ---------------- F. mechanism: what the rewrite contains ----------------
def rewrites(m):
    out = {}
    for line in open(os.path.join(ROOT, "ns_results_v2", f"results_NS_rewrite_{m}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        out[r["question_id"]] = {p: r["personas_eval"][p].get("rewrite") or "" for p in ["alpha", "beta", "gamma"]}
    return out


AGE = re.compile(r"\b\d{1,3}[\s-]*(?:year|month|week|day)s?[\s-]*old\b|\baged?\s+\d{1,3}\b|\b\d{1,3}\s*(?:yo|y/o)\b", re.I)
print("\n=== Rewrite properties (all 12 models) ===")
RP = []
for m in MODELS:
    rw = rewrites(m)
    for q, d in rw.items():
        for p in ["alpha", "beta", "gamma"]:
            t = d[p]
            RP.append(dict(model=m, qid=q, persona=p, words=len(t.split()), age=bool(AGE.search(t)),
                           dx_leak=bool(re.search(r"\b(most likely|consistent with|suggestive of|diagnosis (is|of))\b", t, re.I)),
                           not_specified=bool(re.search(r"not (specified|provided|mentioned|reported)", t, re.I))))
RP = pd.DataFrame(RP); RP.to_csv(os.path.join(OUT, "rewrite_properties.csv"), index=False)
print(RP[RP.model.isin(M11)].groupby("persona")[["words", "age", "dx_leak", "not_specified"]].mean().round(3).to_string())
src_words = {p: DS["persona_" + p].str.split().str.len().mean() for p in ["alpha", "beta", "gamma"]}
print("source narrative words:", {k: round(v) for k, v in src_words.items()}, "original vignette:", round(DS.question.str.split().str.len().mean()))

# residual gap by whether the rewrite states the age, per model, pooled
W = RP[RP.model.isin(M11)].pivot_table(index=["model", "qid"], columns="persona", values="age").reset_index()
rows = []
for m in M11:
    x = arr("NS-rewrite", m).reset_index()
    rows.append(x.assign(model=m).merge(W[W.model == m], on=["model", "qid"], suffixes=("", "_age")))
X = pd.concat(rows)
X["gap"] = X.alpha.astype(float) - (X.beta.astype(float) + X.gamma.astype(float)) / 2
X["age_both"] = X.beta_age.astype(bool) & X.gamma_age.astype(bool)
print("rewrite gap when the rewrites of both marked voices state the age: "
      f"{X[X.age_both].gap.mean():.3f} (n={X.age_both.sum()}); when at least one omits it: {X[~X.age_both].gap.mean():.3f} (n={(~X.age_both).sum()})")
# does rewrite age track source age?
src = A[["beta", "gamma"]].rename(columns={"beta": "src_beta", "gamma": "src_gamma"})
Y = W.merge(src, left_on="qid", right_index=True)
for p in ["beta", "gamma"]:
    print(f"  {p}: rewrite states age in {Y[p].mean():.3f}; when source states it {Y[Y['src_'+p]][p].mean():.3f}, when source does not {Y[~Y['src_'+p]][p].mean():.3f}")

# ---------------- G. oracle (leaky) reference ----------------
o = summary("LEAK-agent1only", M11)
print(f"\nOracle reference (reasoner saw JSON + original clinical vignette): a {o['a']:.3f} marked {o['marked']:.3f} overall {o['overall']:.3f} gap {o['gap']:.3f}")

# ---------------- H. latency ----------------
print("\n=== Latency (s per question, 3 voices; mean over 11 models) ===")
for c in ["agent1only", "notool", "full", "storyform", "rewrite"]:
    v = []
    for m in M11:
        f = os.path.join(ROOT, "ns_results_v2", f"results_NS_{c}_{m}.jsonl")
        if not os.path.exists(f): continue
        t = [sum(json.loads(l)["personas_eval"][p].get("pipeline_latency_sec", 0) for p in ["alpha", "beta", "gamma"]) for l in open(f, encoding="utf-8")]
        if len(t) >= 1000: v.append(np.mean(t))
    print(f"  {c:12s} {np.mean(v):.1f}  (n models {len(v)})")
print("\nNote: latency for agent1only/notool/full v2 excludes Agent-1 extraction (reused from the earlier run); add the earlier Agent-1 time when reporting.")
