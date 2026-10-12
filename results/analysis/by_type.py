import sys, re, numpy as np, pandas as pd
sys.path.insert(0, "."); from load import load_all, MODELS
D = load_all(); M11 = [m for m in MODELS if m != "openbiollm_8b"]
DS = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "medqa_1000.csv")).set_index("question_id")
def qtype(q):
    s = q.lower()[-250:]
    if re.search(r"diagnos|cause of|etiology|most likely (underlying|explanation)|condition", s): return "diagnosis"
    if re.search(r"next step|management|treatment|therapy|pharmacotherapy|medication|intervention|appropriate", s): return "management"
    if re.search(r"test|study|evaluation|confirm|workup|investigat", s): return "investigation"
    return "other"
T = DS.question.map(qtype); print(T.value_counts().to_dict()); print(DS.meta_info.value_counts().to_dict())
rng = np.random.default_rng(0)
def qg(c):
    return pd.concat([(lambda x: x.alpha.astype(float)-(x.beta.astype(float)+x.gamma.astype(float))/2)(D[(c,m)].set_index("qid")) for m in M11], axis=1).mean(axis=1)
for grp, s in [("type", T), ("step", DS.meta_info)]:
    rows = []
    for c in ["B1", "B4", "NS-rewrite"]:
        G = qg(c)
        for k in s.unique():
            v = G[s[s == k].index].values; bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            rows.append(dict(cond=c, group=k, n=len(v), gap=round(v.mean(), 3), ci=f"[{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]"))
    print(pd.DataFrame(rows).pivot(index="group", columns="cond", values="gap").to_string())
    pd.DataFrame(rows).to_csv(f"final/gap_by_{grp}.csv", index=False)
