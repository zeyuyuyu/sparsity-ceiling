"""Aggregate the copy-task row into an M-dependence table.

The copy task exposes an explicit memory load M = (L-1)/2 symbols, which is the
quantity the firing-floor bound rho >= H_b^-1(log2 M / H) is stated in.  Groups
by (variant, L, theta); prints quality + activity per M, then the M-slope of
activity per variant.  Pure re-read of logged JSON, no training.
"""
import json, glob, os, sys, math
from collections import defaultdict

RUNS = "/work/zeyuwang/neuro_poc/ssm3way_runs"
def mean(x): return sum(x) / len(x)
def sd(x):
    if len(x) < 2: return float("nan")
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
def ms(x, p=4):
    return "%.*f +- %.*f" % (p, mean(x), p, sd(x)) if len(x) > 1 else "%.*f (n=1)" % (p, mean(x))

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
    r = json.load(open(f))
    if r.get("task") != "copy": continue
    th = r["analog"]["theta"] if r.get("analog") else None
    groups[(r["variant"], r["L"], th)].append(r)

order = {"digital": 0, "spikeout": 1, "analog": 2, "spikestate": 3}
keys = sorted(groups, key=lambda k: (k[1], order.get(k[0], 9), k[2] or 0))
CHANCE = 1.0 / 16

print("COPY TASK  (K=16 alphabet, chance acc = %.4f, loss/metric on recalled half only)" % CHANCE)
print()
print("| M = (L-1)/2 | variant | n seeds | acc | bpc | rate_emitted | rate_state | E pJ/tok (cons) |")
print("|---|---|---|---|---|---|---|---|")
for k in keys:
    rs = sorted(groups[k], key=lambda r: r["seed"])
    name = k[0] + ("" if k[2] is None else " th=%g" % k[2])
    print("| %d | `%s` | %d (%s) | %s | %s | %s | %s | %s |" % (
        (k[1] - 1) // 2, name, len(rs), ",".join(str(r["seed"]) for r in rs),
        ms([r["acc"] for r in rs]), ms([r["bpc"] for r in rs]),
        ms([r["rate_emitted"] for r in rs]), ms([r["rate_state"] for r in rs]),
        ms([r["energy_pJ_per_token_conservative"] for r in rs], 0)))

print()
print("M-dependence of activity (the bound's prediction: state-in-spikes activity RISES with M):")
for v in ("digital", "spikeout", "analog", "spikestate"):
    row = []
    for k in sorted([k for k in groups if k[0] == v and k[2] in (None, 1.0)], key=lambda k: k[1]):
        rs = groups[k]
        M = (k[1] - 1) // 2
        row.append((M, mean([r["rate_state"] for r in rs]), mean([r["rate_emitted"] for r in rs])))
    if not row: continue
    print("  %-11s " % v + "  ".join("M=%d: state %.3f / emit %.3f" % t for t in row))

print()
print("Paired acc/bpc delta vs digital at the same (L, seed):")
for L in sorted({k[1] for k in groups}):
    base_a = {r["seed"]: r["acc"] for r in groups.get(("digital", L, None), [])}
    base_b = {r["seed"]: r["bpc"] for r in groups.get(("digital", L, None), [])}
    if not base_a: continue
    for k in sorted([k for k in groups if k[1] == L and k[0] != "digital"], key=lambda k: (order.get(k[0], 9), k[2] or 0)):
        rs = [r for r in sorted(groups[k], key=lambda r: r["seed"]) if r["seed"] in base_a]
        if len(rs) < 2: continue
        da = [r["acc"] - base_a[r["seed"]] for r in rs]
        db = [r["bpc"] - base_b[r["seed"]] for r in rs]
        name = k[0] + ("" if k[2] is None else " th=%g" % k[2])
        print("  M=%-3d %-14s dacc %+.4f (sd %.4f)  dbpc %+.4f (sd %.4f)  n=%d" % (
            (L - 1) // 2, name, mean(da), sd(da), mean(db), sd(db), len(rs)))
