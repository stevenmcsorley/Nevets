import json, os, platform, shutil, subprocess, sys

def cmd(args):
    try: return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True,timeout=10).strip()
    except Exception: return None

info={"python":sys.version,"platform":platform.platform(),"executable":sys.executable}
info["nvidia_smi"]=cmd(["nvidia-smi","--query-gpu=name,memory.total,driver_version,compute_cap","--format=csv,noheader"])
try:
 import torch
 info.update(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda)
 if torch.cuda.is_available():
  info["gpu"]=torch.cuda.get_device_name(0)
  info["vram_gb"]=round(torch.cuda.get_device_properties(0).total_memory/2**30,2)
  info["capability"]=".".join(map(str,torch.cuda.get_device_capability(0)))
  x=torch.randn(2048,2048,device="cuda",dtype=torch.float16); y=x@x
  info["cuda_smoke"]=float(y[0,0].item())
except Exception as e: info["torch_error"]=repr(e)
print(json.dumps(info,indent=2))
if info.get("nvidia_smi") and not info.get("cuda_available"):
 print("\nERROR: NVIDIA GPU is visible but PyTorch CUDA is not. Re-run bootstrap with the CUDA wheel.")
 sys.exit(2)
if info.get("vram_gb",0) >= 40: profile="s1-70m (large batches; later try 150-300M)"
elif info.get("vram_gb",0) >= 20: profile="s1-70m"
elif info.get("vram_gb",0) >= 10: profile="s1-35m first; s1-70m with small batch/accumulation"
else: profile="smoke only / reduce model size"
print(f"\nRecommended initial profile: {profile}")
