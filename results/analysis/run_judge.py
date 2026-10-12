import os,sys,re
model=sys.argv[1]
root=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(root)
for l in open(".env"):
    m=re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$",l)
    if m: os.environ[m.group(1)]=m.group(2).strip("'\"")
out=os.path.join(root, "results", "analysis", "judge_"+model)
os.makedirs(out,exist_ok=True)
os.environ.update(AUDIT_MODEL=model,DATASET_CSV=os.path.abspath("dataset/medqa_1000.csv"),OUTPUT_DIR=out)
sys.path.insert(0,"codes"); sys.argv=["x","--workers",os.environ.get("W","40")]
import audit_added_facts as a; a.main()
