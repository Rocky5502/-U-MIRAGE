import argparse, json
from pathlib import Path
from umirage.data import REGISTRY

def main():
 p=argparse.ArgumentParser();p.add_argument("dataset",choices=REGISTRY);p.add_argument("--output",default="data/cache");a=p.parse_args();spec=REGISTRY[a.dataset]
 if "hf_id" not in spec: raise SystemExit(f"{a.dataset}: use official upstream preparation instructions; media is not auto-vendored")
 try: from datasets import load_dataset
 except ImportError as e: raise SystemExit("pip install -e .[data]") from e
 ds=load_dataset(spec["hf_id"]); out=Path(a.output)/a.dataset; out.mkdir(parents=True,exist_ok=True)
 (out/"manifest.json").write_text(json.dumps({"id":a.dataset,"hf_id":spec["hf_id"],"splits":{k:len(v) for k,v in ds.items()}},indent=2))
 print(out)
if __name__=="__main__":main()
