"""
Faithfulness audit of Rewrite-then-Reason rewrites (read-only, LLM judge, no GPU).
For a fixed random sample of questions (same for every model), the judge compares the
patient narrative with the model's clinical rewrite and reports:
  - clinical facts in the narrative that the rewrite omits (social/emotional content the
    rewrite is told to drop is NOT counted as an omission)
  - content in the rewrite that the narrative does not state or directly imply (invented)
  - which invented items are clinically relevant
Output: rewrite_audit.csv (resume-safe), rewrite_audit_summary.csv

Usage: python audit_rewrites.py --results-dir <ns_results_v2> --out-dir <dir> --n 100 --workers 32
Needs GEMINI_API_KEY (or a .env next to the repo root with GEMINI_API_KEY=...).
"""
import argparse, json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

MODEL = os.environ.get("AUDIT_MODEL", "gemini-3.8-flash")
PERSONAS = ["alpha", "beta", "gamma"]


def prompt(narrative, rewrite):
    return f"""You are a clinical auditor. A patient's own account was rewritten by an AI as a clinical case vignette.
The rewrite was instructed to KEEP every clinical fact (age, sex, symptoms and denied symptoms, onset/duration/course, all numbers, history, medications, remedies or substances taken, exposures, exam/lab/imaging findings) and to DROP non-clinical content (finances, work or family pressure, who brought the patient, emotions, metaphor).

PATIENT NARRATIVE:
{narrative}

REWRITTEN VIGNETTE:
{rewrite}

1. List each CLINICAL fact in the narrative that is missing from or changed in the rewrite. Do not count dropped non-clinical content.
2. List each item in the rewrite that the narrative does not state or directly imply (invented content), and mark whether it is clinically relevant (could change diagnosis or management).
3. Return ONLY one line of JSON, no markdown:
{{"clinical_facts_in_narrative":<int>,"missing":["<5 words>"],"invented":[{{"item":"<5 words>","clinically_relevant":<true|false>}}]}}"""


def parse(raw):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", raw or ""), re.DOTALL)
    if not m:
        return None
    try:
        r = json.loads(m.group(0))
        return {"n_facts": int(r.get("clinical_facts_in_narrative", 0)), "missing": r.get("missing", []),
                "invented": r.get("invented", [])}
    except Exception:
        return None


_m = None
def judge(p):
    global _m
    import google.generativeai as genai
    if _m is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _m = genai.GenerativeModel(MODEL, generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=8192))
    for a in range(4):
        try:
            r = parse(_m.generate_content(p).text)
            if r:
                return r
        except Exception:
            time.sleep(5 * (a + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        for env in [Path(a.dataset).resolve().parents[1] / ".env"]:
            if env.exists():
                for l in open(env):
                    mm = re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$", l)
                    if mm:
                        os.environ[mm.group(1)] = mm.group(2).strip("'\"")
    ds = pd.read_csv(a.dataset).set_index("question_id")
    qids = sorted(random.Random(7).sample(sorted(ds.index), a.n))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "rewrite_audit.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.model, prev.question_id, prev.persona))
    jobs = []
    for f in sorted(Path(a.results_dir).glob("results_NS_rewrite_*.jsonl")):
        model = f.stem[len("results_NS_rewrite_"):]
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["question_id"] not in qids:
                continue
            for p in PERSONAS:
                if (model, r["question_id"], p) in done:
                    continue
                rw = r["personas_eval"][p].get("rewrite") or ""
                jobs.append((model, r["question_id"], p, ds.loc[r["question_id"], f"persona_{p}"], rw))
    print(f"{len(jobs)} audits to run")

    def work(j):
        model, q, p, nar, rw = j
        res = judge(prompt(nar, rw)) if rw.strip() else {"n_facts": -1, "missing": [], "invented": []}
        if res is None:
            return None
        inv = res["invented"] if isinstance(res["invented"], list) else []
        return {"model": model, "question_id": q, "persona": p, "n_facts": res["n_facts"],
                "n_missing": len(res["missing"]), "n_invented": len(inv),
                "n_invented_relevant": sum(bool(x.get("clinically_relevant")) for x in inv if isinstance(x, dict)),
                "rewrite_words": len(rw.split()), "missing": json.dumps(res["missing"]), "invented": json.dumps(inv)}

    header = not path.exists()
    with ThreadPoolExecutor(a.workers) as ex, open(path, "a", newline="", encoding="utf-8") as fh:
        for i, fut in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            row = fut.result()
            if row:
                pd.DataFrame([row]).to_csv(fh, header=header, index=False); fh.flush(); header = False
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    d = pd.read_csv(path).drop_duplicates(["model", "question_id", "persona"], keep="last")
    d = d[d.n_facts > 0]
    d["recall"] = 1 - (d.n_missing / d.n_facts).clip(upper=1)
    s = d.groupby(["model", "persona"]).agg(n=("recall", "size"), recall=("recall", "mean"),
                                           missing=("n_missing", "mean"), invented=("n_invented", "mean"),
                                           invented_relevant=("n_invented_relevant", "mean"),
                                           pct_any_relevant_invented=("n_invented_relevant", lambda x: (x > 0).mean()))
    s.round(3).to_csv(out / "rewrite_audit_summary.csv")
    print(s.groupby("persona").mean().round(3).to_string())


if __name__ == "__main__":
    main()
