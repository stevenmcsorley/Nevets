import argparse
from systemone_lab.data.spatial_worlds import generate_counterfactual_jsonl
ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--pairs',type=int,default=2000); ap.add_argument('--seed',type=int,default=4401); ap.add_argument('--min-hops',type=int,default=2); ap.add_argument('--max-hops',type=int,default=8); a=ap.parse_args(); generate_counterfactual_jsonl(a.out,a.pairs,a.seed,a.min_hops,a.max_hops); print(a.out)
