"""Aggregate the FashionMNIST row-stream row (V=10, the small-output shape).

Usage: python agg_stream.py [epochs]        (default 10)

Reports accuracy (chance 0.10), margin kept vs the NOISE-REGULARIZED digital reference,
measured event rates, and the 45nm proxy pJ/token ratio vs that same reference.
Groups by (variant, theta, dig_noise, in_theta, out_theta, epochs) so budgets and gate
settings are never pooled -- the bug class of agg_copy.py (34e1fd0) and
energy_datapath.py's label() (d47511b).
"""
import json, glob, sys, statistics as st
from collections import defaultdict

EP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CHANCE = 0.10

recs = defaultdict(list)
for f in sorted(glob.glob("ssm3way_runs/*stream*.json")):
    r = json.load(open(f))
    if r.get("task") != "stream" or r.get("epochs") != EP:
        continue
    key = (r["variant"],
           r["analog"]["theta"] if r["analog"] else None,
           r["dig_reg"]["dig_noise"], r.get("in_theta", 0.0), r.get("out_theta", 0.0))
    recs[key].append(r)

def mean(v):
    return st.mean(v) if v else float("nan")
def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0

ref_key = ("digital", None, 0.02, 0.0, 0.0)
if ref_key not in recs:
    print("no noise-regularized digital reference at this budget"); sys.exit(1)
ref_acc = mean([r["acc"] for r in recs[ref_key]])
ref_pj = mean([r["energy_pJ_per_token"] for r in recs[ref_key]])

def label(k):
    v, th, dn, it, ot = k
    s = v
    if th is not None: s += f" th={th}"
    if dn > 0: s += " +n0.02"
    if it > 0: s += f" in_th={it}"
    if ot > 0: s += f" out_th={ot}"
    return s

print(f"FashionMNIST row-stream, V=10, ep{EP}.  chance acc {CHANCE}; "
      f"reference = noise-regularized digital, acc {ref_acc:.4f}, {ref_pj:.0f} pJ/token\n")
hdr = ("cell", "n", "acc", "sd", "margin", "bpc", "r_z", "r_s", "r_in", "r_out",
       "pJ/tok", "pJ_cons", "x_vs_ref")
print("{:26s} {:>2s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>9s} {:>9s} {:>8s}".format(*hdr))
for k in sorted(recs, key=lambda k: (k[0], k[1] or 0, k[3], k[4])):
    rs = recs[k]
    acc = mean([r["acc"] for r in rs])
    row = [label(k), len(rs), acc, sd([r["acc"] for r in rs]),
           (acc - CHANCE) / (ref_acc - CHANCE),
           mean([r["bpc"] for r in rs]), mean([r["rate_emitted"] for r in rs]),
           mean([r["rate_state"] for r in rs]), mean([r.get("rate_in", 1.0) for r in rs]),
           mean([r["rate_out"] for r in rs]),
           mean([r["energy_pJ_per_token"] for r in rs]),
           mean([r["energy_pJ_per_token_conservative"] for r in rs])]
    row.append(ref_pj / row[10])
    print("{:26s} {:2d} {:7.4f} {:6.4f} {:6.3f} {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.4f} "
          "{:9.0f} {:9.0f} {:8.3f}".format(*row))

print("\nPRE-REGISTERED CRITERIA (d47511b / 3033631), applied mechanically:")
best = None
for k, rs in recs.items():
    if k[0] != "analog":
        continue
    acc = mean([r["acc"] for r in rs]); rz = mean([r["rate_emitted"] for r in rs])
    keep = acc / ref_acc
    ok = keep >= 0.95 and rz <= 0.65
    print(f"  th={k[1]}: acc/ref {keep:.4f} at r_z {rz:.4f}  "
          f"-> CONFIRM cell {'YES' if ok else 'no'}"
          f"   (deficit {100*(ref_acc-acc):.2f} pts, pJ x{ref_pj/mean([r['energy_pJ_per_token'] for r in rs]):.2f})")
    if ok and (best is None or rz < best[1]):
        best = (k[1], rz, acc)
refute = all(mean([r["acc"] for r in rs]) < ref_acc - 0.05
             for k, rs in recs.items()
             if k[0] == "analog" and mean([r["rate_emitted"] for r in rs]) <= 0.8)
n_usable = sum(1 for k, rs in recs.items()
               if k[0] == "analog" and mean([r["rate_emitted"] for r in rs]) <= 0.8)
print(f"\n  CONFIRM (some theta keeps >=0.95 of ref acc at r_z<=0.65): "
      f"{'MET at theta='+str(best[0]) if best else 'NOT met'}")
print(f"  REFUTE  (>5-pt deficit at EVERY theta with r_z<=0.8): "
      f"{'MET' if refute and n_usable else 'not met'}  ({n_usable} cells with r_z<=0.8)")
