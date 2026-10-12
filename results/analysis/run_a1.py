import os,sys,re
root=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(root)
for l in open(".env"):
    m=re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$",l)
    if m: os.environ[m.group(1)]=m.group(2).strip("'\"")
os.environ.update(AUDIT_MODEL="gemini-3.8-flash",AUDIT_WORKERS="16",DATASET_CSV=os.path.abspath("dataset/medqa_1000.csv"),
 NS_RESULT_DIR=os.path.join(root, "results", "ns_results_v2"),OUTPUT_DIR=os.path.join(root, "results", "analysis", "agent1_audit"))
sys.path.insert(0,"codes"); sys.argv=["x","--sample-size","100","--ablation","full"]
import audit_agent1_outputs as a; a.main()
