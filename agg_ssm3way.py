"""Aggregate ssm3way per-run JSON into 3-seed mean+sd tables (markdown).

Usage: python agg_ssm3way.py [task]   (default task=charlm)
Reads /work/zeyuwang/neuro_poc/ssm3way_runs/*.json, groups by (variant, task,
analog theta), prints a markdown table plus paired per-seed deltas vs digital.
Deterministic: no training, pure re-read of the logged JSON.
"""
import json, glob, os, sys, math
from collections import defaultdict

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssm3way_runs")
task = sys.argv[1] if len(sys.argv) > 1 else "charlm"

def mean(x): return sum(x) / len(x)
def sd(x):
    if len(x) < 2: return float("nan")
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
def ms(x, p=4):
    return "%.*f +- %.*f" % (p, mean(x), p, sd(x)) if len(x) > 1 else "%.*f (n=1)" % (p, mean(x))

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
    r = json.load(open(f))
    if r.get("task") != task: continue
    th = r["analog"]["theta"] if r.get("analog") else None
    key = (r["variant"], th)
    groups[key].append(r)

order = {"digital": 0, "spikeout": 1, "analog": 2, "spikestate": 3}
keys = sorted(groups, key=lambda k: (order.get(k[0], 9), k[1] if k[1] is not None else 0))

print("task=%s   cells=%d" % (task, len(keys)))
print()
print("| variant | n seeds | bpc | acc | rate_emitted | rate_state | E pJ/tok (cons) |")
print("|---|---|---|---|---|---|---|")
for k in keys:
    rs = sorted(groups[k], key=lambda r: r["seed"])
    name = k[0] + ("" if k[1] is None else " th=%g" % k[1])
    seeds = ",".join(str(r["seed"]) for r in rs)
    print("| `%s` | %d (%s) | %s | %s | %s | %s | %s |" % (
        name, len(rs), seeds,
        ms([r["bpc"] for r in rs]), ms([r["acc"] for r in rs]),
        ms([r["rate_emitted"] for r in rs]), ms([r["rate_state"] for r in rs]),
        ms([r["energy_pJ_per_token_conservative"] for r in rs], 0)))

# paired per-seed delta vs digital (same seed) -- stronger than comparing means
base = {r["seed"]: r["bpc"] for r in groups.get(("digital", None), [])}
if base:
    print()
    print("Paired bpc delta vs digital, same seed (positive = worse than digital):")
    for k in keys:
        if k[0] == "digital": continue
        d = [(r["seed"], round(r["bpc"] - base[r["seed"]], 4)) for r in sorted(groups[k], key=lambda r: r["seed"]) if r["seed"] in base]
        if len(d) < 2: continue
        vals = [v for _, v in d]
        name = k[0] + ("" if k[1] is None else " th=%g" % k[1])
        print("  %-16s per-seed %s  mean %+.4f  sd %.4f  all-same-sign=%s" % (
            name, d, mean(vals), sd(vals), all(v > 0 for v in vals) or all(v < 0 for v in vals)))
