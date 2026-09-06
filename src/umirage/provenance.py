from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from pathlib import Path

def _cmd(args):
    try:return subprocess.check_output(args,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def manifest(extra=None):
    m={"python":sys.version.split()[0],"platform":platform.platform(),"git_sha":_cmd(["git","rev-parse","HEAD"]),"nvidia_smi":_cmd(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"])}
    try:
        import torch; m|={"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available()}
    except Exception:m["torch"]=None
    if extra:m.update(extra)
    return m

def hash_text(s):return hashlib.sha256(s.encode()).hexdigest()
def write(path,extra=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(manifest(extra),indent=2,sort_keys=True))
