import json, numpy as np, pandas as pd
from load import ROOT, P, MODELS
CH = [m for m in MODELS if m != "openbiollm_8b"]
def rd(abl, m):
    d = {}
    for l in open(f"{ROOT}/ns_results/results_NS_{abl}_{m}.jsonl", encoding="utf-8"):
        r = json.loads(l); d[r["question_id"]] = {p: (r["personas_eval"][p]["is_correct"], r["personas_eval"][p].get("tools_called", 0)) for p in P}
    return d
tot_used = []; rows = []
for m in CH:
    f = rd("full", m); n = rd("notool", m)
    used_f, used_n, ua, un, ma = [], [], [], [], []
    for q in f:
        for p in P:
            cf, t = f[q][p]; cn, _ = n[q][p]
            (used_f if t > 0 else ma).append(cf); (used_n if t > 0 else []).append(cn)
            if t == 0: un.append(cn)
    rows.append(dict(model=m, n_with_tools=len(used_f), pct_tools=100*len(used_f)/(len(used_f)+len(ma)),
                     acc_full_tools=np.mean(used_f), acc_notool_same_items=np.mean(used_n),
                     acc_full_no_tools=np.mean(ma), acc_notool_same_items_no_tools=np.mean(un)))
R = pd.DataFrame(rows)
print(R.round(3).to_string(index=False))
d = (R.acc_full_tools - R.acc_notool_same_items)
print("\nOn items where tools fired: full - notool accuracy = %+.3f (mean over 11 models; positive = tools help)" % d.mean())
print("models where tools help: %d / 11" % (d > 0).sum())
d0 = R.acc_full_no_tools - R.acc_notool_same_items_no_tools
print("On items where NO tool fired (pure run-to-run noise between full and notool): %+.3f" % d0.mean())
