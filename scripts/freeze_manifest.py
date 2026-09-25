import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
import torch
ap=argparse.ArgumentParser(); ap.add_argument('--file',action='append',default=[]); ap.add_argument('--out',default='reports/run_manifest.json'); a=ap.parse_args()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def cmd(c):
 try:return subprocess.check_output(c,text=True,stderr=subprocess.DEVNULL).strip()
 except:return None
m={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'git_commit':cmd(['git','rev-parse','HEAD']),'files':{}}
if torch.cuda.is_available(): m['gpu']={'name':torch.cuda.get_device_name(0),'vram':torch.cuda.get_device_properties(0).total_memory,'capability':torch.cuda.get_device_capability(0)}
for p in a.file:
 pp=Path(p); m['files'][str(pp)]={'sha256':sha(pp),'bytes':pp.stat().st_size}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(m,indent=2)); print(out)
