import os,sys,re
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
for l in open(".env"):
    m=re.match(r"\s*(\w+)\s*=\s*(.*?)\s*$",l)
    if m: os.environ[m.group(1)]=m.group(2).strip("'\"")
os.environ.update(AUDIT_MODEL="gemini-3.8-flash",DATASET_CSV=os.path.abspath("dataset/medqa_1000.csv"),
 OUTPUT_DIR=os.path.join(os.getcwd(), "results", "analysis"))
sys.path.insert(0,"codes"); sys.argv=["x","--workers","32"]
import audit_added_facts as a; a.main()
