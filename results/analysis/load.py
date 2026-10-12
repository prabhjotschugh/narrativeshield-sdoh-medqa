import json, glob, os
import numpy as np, pandas as pd

ROOT = os.environ.get("NS_RESULTS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
P = ["alpha", "beta", "gamma"]
MODELS = ["llama32_3b","medgemma_4b","phi4_mini","qwen25_7b","mistral7b","qwen3_8b","ministral8b",
          "llama31_8b","openbiollm_8b","gemma3_12b","qwen25_14b","mistral_small_24b"]
SIZE = dict(zip(MODELS, [3,4,4,7,7,8,8,8,8,12,14,24]))

def load_all(include_leaky=False):
    """returns dict[(cond, model)] -> DataFrame(question_id, alpha, beta, gamma) of is_correct (bool), plus parsed flags"""
    out = {}
    for cond in ["b1","b2","b3","b4"]:
        for f in glob.glob(f"{ROOT}/{cond}_results/results_{cond.upper()}_*.jsonl"):
            m = os.path.basename(f)[len(f"results_{cond.upper()}_"):-6]
            out[(cond.upper(), m)] = read(f, ans="extracted_answer")
    # Valid structured-pipeline results (vignette leak fixed). Partial files (<1000 rows) are skipped.
    for f in glob.glob(f"{ROOT}/ns_results_v2/results_NS_*_*.jsonl"):
        b = os.path.basename(f)[len("results_NS_"):-6]
        abl, m = b.split("_", 1)
        df = read(f, ans="final_answer")
        if len(df) >= 1000:
            out[("NS-"+abl, m)] = df
    # Old runs whose reasoning call saw the original vignette (bug). Kept only as an oracle reference.
    if include_leaky:
        for f in glob.glob(f"{ROOT}/ns_results/results_NS_*_*.jsonl"):
            b = os.path.basename(f)[len("results_NS_"):-6]
            abl, m = b.split("_", 1)
            out[("LEAK-"+abl, m)] = read(f, ans="final_answer")
    return out

def read(f, ans):
    rows = []
    for l in open(f, encoding="utf-8"):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        d = {"qid": r["question_id"]}
        for p in P:
            e = r["personas_eval"][p]
            d[p] = bool(e["is_correct"])
            d[p+"_parsed"] = e.get(ans) is not None
            d[p+"_ans"] = e.get(ans)
        rows.append(d)
    return pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)

if __name__ == "__main__":
    D = load_all()
    print(len(D), "result sets")
    for k in sorted(D)[:5]:
        print(k, len(D[k]))
