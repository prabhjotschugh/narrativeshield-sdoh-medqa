"""
Quick look at the NarrativeShield v2 reruns (ns_results_v2) while they run.
Standard library only, so it runs on the login node:
    python check_v2.py                 # all ablations
    python check_v2.py agent1only      # one ablation
Partial files are summarised on the questions done so far, and B1 is
restricted to the same questions so the comparison is paired.
"""
import json, os, sys

BASE = os.environ.get("SCRATCH_BASE", "/path/to/scratch")
V2, OLD, B1 = f"{BASE}/ns_results_v2", f"{BASE}/ns_results", f"{BASE}/b1_results"
MODELS = ["llama32_3b", "medgemma_4b", "phi4_mini", "qwen25_7b", "mistral7b", "qwen3_8b", "ministral8b",
          "llama31_8b", "openbiollm_8b", "gemma3_12b", "qwen25_14b", "mistral_small_24b"]
P = ["alpha", "beta", "gamma"]
ABL = sys.argv[1:] or ["agent1only", "notool", "full"]


def load(path, ans_key):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            e = r["personas_eval"]
            out[r["question_id"]] = {p: (bool(e[p]["is_correct"]), e[p].get(ans_key) is not None,
                                         bool(e[p].get("agent1_parse_success", True))) for p in P}
    return out


def summ(d, qids):
    n = len(qids)
    acc = {p: sum(d[q][p][0] for q in qids) / n for p in P}
    parse = sum(d[q][p][1] for q in qids for p in P) / (3 * n)
    gap = acc["alpha"] - (acc["beta"] + acc["gamma"]) / 2
    nag = acc["alpha"] - min(acc["beta"], acc["gamma"])
    return acc, gap, nag, parse


for abl in ABL:
    print(f"\n=== {abl}  (v2 = fixed rerun; B1 and old-leaky on the same questions) ===")
    print(f"{'model':18s} {'n':>5s} | {'v2 a/b/g':>17s} {'gap':>6s} {'NAG':>6s} {'parse':>5s} | "
          f"{'B1 a/b/g':>17s} {'gap':>6s} | {'old gap':>7s}")
    agg = []
    for m in MODELS:
        v2 = load(f"{V2}/results_NS_{abl}_{m}.jsonl", "final_answer")
        if not v2:
            print(f"{m:18s} {'-':>5s} | not started")
            continue
        b1 = load(f"{B1}/results_B1_{m}.jsonl", "extracted_answer")
        old = load(f"{OLD}/results_NS_{abl}_{m}.jsonl", "final_answer")
        q = sorted(set(v2) & set(b1))
        if not q:
            continue
        a, g, nag, pr = summ(v2, q)
        ba, bg, _, _ = summ(b1, q)
        qo = [x for x in q if x in old]
        og = summ(old, qo)[1] if qo else float("nan")
        flag = "" if len(v2) >= 1000 else "  (partial)"
        print(f"{m:18s} {len(q):5d} | {a['alpha']:.3f}/{a['beta']:.3f}/{a['gamma']:.3f} {g:+.3f} {nag:+.3f} {pr:5.2f} | "
              f"{ba['alpha']:.3f}/{ba['beta']:.3f}/{ba['gamma']:.3f} {bg:+.3f} | {og:+7.3f}{flag}")
        if m != "openbiollm_8b":
            agg.append((a, g, ba, bg, og))
    if agg:
        k = len(agg)
        mean = lambda f: sum(f(x) for x in agg) / k
        print(f"{'MEAN (excl OpenBio)':18s} {k:5d} | "
              f"{mean(lambda x: x[0]['alpha']):.3f}/{mean(lambda x: x[0]['beta']):.3f}/{mean(lambda x: x[0]['gamma']):.3f} "
              f"{mean(lambda x: x[1]):+.3f} {'':6s} {'':5s} | "
              f"{mean(lambda x: x[2]['alpha']):.3f}/{mean(lambda x: x[2]['beta']):.3f}/{mean(lambda x: x[2]['gamma']):.3f} "
              f"{mean(lambda x: x[3]):+.3f} | {mean(lambda x: x[4]):+7.3f}")
        print("gap = acc(alpha) - mean(acc(beta), acc(gamma)); null NAG 95th pct ~0.027 at n=1000 (wider for partial files)")
