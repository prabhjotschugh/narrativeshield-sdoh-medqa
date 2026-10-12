"""
Added-facts audit (reverse direction of the dataset fact-preservation audit).
=============================================================================
The dataset audit checks that every ORIGINAL fact survives in a persona
rewrite. A complementary question: the opposite: does a marked
persona (beta / gamma) ADD clinically relevant information that the original
question never contained? If so, part of the gap could come from new
content, not register.

READ-ONLY: this script never modifies dataset/ or any persona text. It only
reads the personas, asks an independent LLM judge to list material that is in
the rewrite but not in the original, and writes a CSV next to your results.

Output: added_facts_audit.csv  (one row per question x persona; resume-safe)
        added_facts_summary.csv (rates per persona; the subset of questions
                                 with no clinically relevant additions in
                                 either marked persona is `clean_qids.txt`)

Run (needs GEMINI_API_KEY in the environment; no GPU):
  set DATASET_CSV=<path>\\medqa_1000.csv     (or leave unset to read HF)
  python audit_added_facts.py --personas alpha beta gamma
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

AUDIT_MODEL = os.environ.get("AUDIT_MODEL", "gemini-3.1-flash-lite")
DATASET_NAME = os.environ.get("DATASET_NAME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "medqa_1000.csv"))
DATASET_CSV = os.environ.get("DATASET_CSV")
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
COLS = {"alpha": "persona_alpha", "beta": "persona_beta", "gamma": "persona_gamma"}
MAX_RETRIES = 3


def build_prompt(original: str, rewrite: str, name: str) -> str:
    return f"""You are a clinical auditor for a medical NLP study.

TASK: Find information in the REWRITTEN narrative that is NOT in the ORIGINAL case.

ORIGINAL:
{original}

REWRITTEN ({name}):
{rewrite}

INSTRUCTIONS:
1. List every discrete piece of information in the REWRITTEN that the ORIGINAL does not contain or imply
   (new symptoms, durations, exposures, behaviours, home remedies, medications, prior care, finances, family context, emotions).
   Ignore pure wording, tone and style changes.
2. For each, set "clinically_relevant" true ONLY if a clinician could use it to choose among diagnoses or management
   (new symptom, exposure, drug/remedy, timeline, risk factor, prior treatment). Social/financial/emotional framing that
   does not bear on diagnosis or management is false.
3. Return ONLY this single-line JSON, no markdown:
{{"added":[{{"item":"<5 words max>","clinically_relevant":<true|false>}}],"n_added":<int>,"n_clinically_relevant":<int>,"note":"<one short sentence>"}}
""".strip()


def parse(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", (raw or "")).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            r = json.loads(m.group(0))
            r.setdefault("added", [])
            r["n_added"] = int(r.get("n_added", len(r["added"])))
            r["n_clinically_relevant"] = int(
                r.get("n_clinically_relevant", sum(bool(a.get("clinically_relevant")) for a in r["added"])))
            return r
        except Exception:
            pass
    return {"error": f"unparseable: {raw[:120]}"}


_model = None
def judge(prompt: str) -> dict:
    global _model
    import google.generativeai as genai
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel(AUDIT_MODEL, generation_config=genai.GenerationConfig(
            temperature=0.0, max_output_tokens=8192, candidate_count=1))
    for a in range(1, MAX_RETRIES + 1):
        try:
            return parse(_model.generate_content(prompt).text)
        except Exception as e:
            if a == MAX_RETRIES:
                return {"error": f"api: {e}"}
            time.sleep(6 * a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", default=["alpha", "beta", "gamma"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY first.")

    if DATASET_CSV:
        df = pd.read_csv(DATASET_CSV)
    else:
        from datasets import load_dataset
        df = load_dataset("csv", data_files=DATASET_NAME, split="train").to_pandas()

    out = OUT_DIR / "added_facts_audit.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.question_id, prev.persona))
    todo = [(r.question_id, p, r.question, r[COLS[p]]) for _, r in df.iterrows()
            for p in args.personas if (r.question_id, p) not in done]
    print(f"{len(todo)} audits to run ({len(done)} already done)")

    def work(t):
        qid, p, q, txt = t
        r = judge(build_prompt(q, txt, f"Persona {p.upper()}"))
        return {"question_id": qid, "persona": p, "n_added": r.get("n_added", -1),
                "n_clinically_relevant": r.get("n_clinically_relevant", -1),
                "items": json.dumps(r.get("added", [])), "note": r.get("note", r.get("error", ""))}

    header = not out.exists()
    with ThreadPoolExecutor(args.workers) as ex, open(out, "a", newline="", encoding="utf-8") as fh:
        for i, f in enumerate(as_completed([ex.submit(work, t) for t in todo]), 1):
            pd.DataFrame([f.result()]).to_csv(fh, header=header, index=False); fh.flush(); header = False
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")

    a = pd.read_csv(out)
    ok = a[a.n_added >= 0]
    s = ok.groupby("persona").agg(n=("question_id", "count"), mean_added=("n_added", "mean"),
                                  mean_relevant=("n_clinically_relevant", "mean"),
                                  pct_any_relevant=("n_clinically_relevant", lambda x: (x > 0).mean()))
    s.to_csv(OUT_DIR / "added_facts_summary.csv"); print(s.round(3).to_string())
    marked = ok[ok.persona.isin(["beta", "gamma"])].groupby("question_id").n_clinically_relevant.max()
    clean = sorted(marked[marked == 0].index)
    (OUT_DIR / "clean_qids.txt").write_text("\n".join(clean))
    print(f"{len(clean)} questions have zero clinically relevant additions in both marked personas")


if __name__ == "__main__":
    main()
