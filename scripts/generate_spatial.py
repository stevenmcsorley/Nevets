import argparse
from systemone_lab.data.spatial_worlds import generate_jsonl
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--n",type=int,default=10000); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--split",default="train"); ap.add_argument("--min-hops",type=int,default=1); ap.add_argument("--max-hops",type=int,default=6)
ap.add_argument("--rename-probability",type=float,default=0.5,help="Fraction using random letter names (both train and dev)")
a=ap.parse_args(); generate_jsonl(a.out,a.n,a.seed,a.split,a.min_hops,a.max_hops,a.rename_probability); print(a.out)
