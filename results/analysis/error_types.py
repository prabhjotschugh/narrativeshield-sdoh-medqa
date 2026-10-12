"""What do models answer when the voice changes the answer? Option-category analysis."""
import ast, re, sys, os, numpy as np, pandas as pd
sys.path.insert(0, "."); from load import load_all, MODELS
D = load_all(); M11 = [m for m in MODELS if m != "openbiollm_8b"]
DS = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "medqa_1000.csv")).set_index("question_id")
OPT = {q: ast.literal_eval(o) for q, o in DS.options.items()}
CAT = {
 "psych/behavioural": r"psych|conversion|somatic|factitious|malinger|anxiety|panic|depress|counsel|cognitive behav|behavio|psychotherap|reassur|stress|hypochondr|illness anxiety|adjustment|personality",
 "minimise/defer": r"reassur|observation|watchful|no (further|additional)|follow[- ]up|discharge|outpatient|lifestyle|rest\b|supportive|routine",
 "social/adherence": r"social work|adherence|nonadherence|noncomplian|financial|insurance|substance|drug[- ]seeking|abuse|withdrawal|alcohol",
}
def cat(text, c): return bool(re.search(CAT[c], text, re.I))
rows = []
for cond in ["B1", "NS-rewrite"]:
    for m in M11:
        x = D[(cond, m)].set_index("qid")
        for q, r in x.iterrows():
            gold = DS.loc[q, "answer_idx"]
            for p in ["alpha", "beta", "gamma"]:
                a = r[p + "_ans"]
                if a is None or a == gold or a not in OPT[q]: continue
                t = OPT[q][a]
                rows.append(dict(cond=cond, model=m, qid=q, persona=p, marked=p != "alpha",
                                 **{c: cat(t, c) and not cat(OPT[q][gold], c) for c in CAT}))
E_ = pd.DataFrame(rows)
print("Share of WRONG answers falling in each category (option not in that category when gold is), 11 models")
print(E_.groupby(["cond", "persona"])[list(CAT)].mean().round(3).to_string())
# per-question flips: alpha correct, marked wrong -> what did the marked voice pick?
F = []
for m in M11:
    x = D[("B1", m)].set_index("qid")
    for q, r in x.iterrows():
        gold = DS.loc[q, "answer_idx"]
        for p in ["beta", "gamma"]:
            if r["alpha"] and not r[p] and r[p + "_ans"] in OPT[q]:
                F.append(dict(model=m, qid=q, persona=p, **{c: cat(OPT[q][r[p+"_ans"]], c) and not cat(OPT[q][gold], c) for c in CAT}))
F = pd.DataFrame(F)
print(f"\nB1 flips (alpha right, marked wrong): n={len(F)}; share landing on category:")
print(F.groupby("persona")[list(CAT)].mean().round(3).to_string())
# base rate: among all wrong options available (not gold), how many are in each category
base = {c: np.mean([cat(t, c) and not cat(OPT[q][DS.loc[q,'answer_idx']], c) for q in DS.index for k, t in OPT[q].items() if k != DS.loc[q,'answer_idx']]) for c in CAT}
print("base rate among all distractors:", {k: round(v, 3) for k, v in base.items()})
E_.to_csv("final/error_categories.csv", index=False); F.to_csv("final/b1_flips_categories.csv", index=False)
