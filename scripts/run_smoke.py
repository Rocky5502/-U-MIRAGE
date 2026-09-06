from umirage.cli import run_smoke
if __name__=="__main__":
 import json; print(json.dumps(run_smoke("artifacts/smoke"),indent=2))
