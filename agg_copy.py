"""Aggregate the copy-task row into an M-dependence table.

The copy task exposes an explicit memory load M = (L-1)/2 symbols, which is the
quantity the firing-floor bound rho >= H_b^-1(log2 M / H) is stated in.  Groups
by (epochs, variant, L, theta); prints quality + activity per M, then the
M-slope of activity per variant.  Pure re-read of logged JSON, no training.

Two runs of the copy task exist at different training budgets (6 epochs /
copy_n 20k, and 30 epochs / copy_n 80k).  The 6-epoch row is INCONCLUSIVE --
every non-digital variant sat at chance -- so `epochs` is part of the group key
and the budgets are never averaged together.  Pass a budget as argv[1] to
restrict the table, e.g. `python agg_copy.py 30`.

The theta calibration wrote analog L=65 s0 cells whose config coincides with a
cell of the main row (theta=0.2, ep30); identical (epochs, variant, L, theta,
seed) configs are de-duplicated so a repeated cell cannot inflate n.
"""
import json, glob, os, sys, math
from collections import defaultdict

RUNS = "/work/zeyuwang/neuro_poc/ssm3way_runs"
WANT_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else None

def mean(x): return sum(x) / len(x)
def sd(x):
    if len(x) < 2: return float("nan")
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
def ms(x, p=4):
    return "%.*f +- %.*f" % (p, mean(x), p, sd(x)) if len(x) > 1 else "%.*f (n=1)" % (p, mean(x))

groups = defaultdict(dict)   # key -> {seed: record}, dict keyed by seed de-dups
for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
    r = json.load(open(f))
    if r.get("task") != "copy": continue
    if WANT_EPOCHS is not None and r["epochs"] != WANT_EPOCHS: continue
    th = r["analog"]["theta"] if r.get("analog") else None
    groups[(r["epochs"], r["variant"], r["L"], th)].setdefault(r["seed"], r)
groups = {k: list(v.values()) for k, v in groups.items()}

order = {"digital": 0, "spikeout": 1, "analog": 2, "spikestate": 3}
keys = sorted(groups, key=lambda k: (k[0], k[2], order.get(k[1], 9), k[3] or 0))
CHANCE = 1.0 / 16

print("COPY TASK  (K=16 alphabet, chance acc = %.4f, loss/metric on recalled half only)" % CHANCE)
print("Budgets are NOT pooled: ep=6 is the inconclusive old row, ep=30 the rerun.")
print()
print("| ep | M = (L-1)/2 | variant | n seeds | acc | bpc | rate_emitted | rate_state | E pJ/tok (cons) |")
print("|---|---|---|---|---|---|---|---|---|")
for k in keys:
    rs = sorted(groups[k], key=lambda r: r["seed"])
    name = k[1] + ("" if k[3] is None else " th=%g" % k[3])
    print("| %d | %d | `%s` | %d (%s) | %s | %s | %s | %s | %s |" % (
        k[0], (k[2] - 1) // 2, name, len(rs), ",".join(str(r["seed"]) for r in rs),
        ms([r["acc"] for r in rs]), ms([r["bpc"] for r in rs]),
        ms([r["rate_emitted"] for r in rs]), ms([r["rate_state"] for r in rs]),
        ms([r["energy_pJ_per_token_conservative"] for r in rs], 0)))

for ep in sorted({k[0] for k in groups}):
    print()
    print("=== budget: %d epochs ===" % ep)
    print("M-dependence of activity (the bound's prediction: state-in-spikes activity RISES with M):")
    for v in ("digital", "spikeout", "analog", "spikestate"):
        row = []
        for k in sorted([k for k in groups if k[0] == ep and k[1] == v], key=lambda k: k[2]):
            rs = groups[k]
            row.append(((k[2] - 1) // 2, mean([r["rate_state"] for r in rs]),
                        mean([r["rate_emitted"] for r in rs]), k[3]))
        if not row: continue
        print("  %-11s " % v + "  ".join(
            "M=%d%s: state %.3f / emit %.3f" % (t[0], "" if t[3] is None else " th=%g" % t[3], t[1], t[2])
            for t in row))

    print()
    print("Paired acc/bpc delta vs digital at the same (L, seed):")
    for L in sorted({k[2] for k in groups if k[0] == ep}):
        base = {r["seed"]: r for r in groups.get((ep, "digital", L, None), [])}
        if not base: continue
        for k in sorted([k for k in groups if k[0] == ep and k[2] == L and k[1] != "digital"],
                        key=lambda k: (order.get(k[1], 9), k[3] or 0)):
            rs = [r for r in sorted(groups[k], key=lambda r: r["seed"]) if r["seed"] in base]
            if not rs: continue
            da = [r["acc"] - base[r["seed"]]["acc"] for r in rs]
            db = [r["bpc"] - base[r["seed"]]["bpc"] for r in rs]
            # fraction of the baseline's above-chance margin that the variant keeps
            keep = [max(0.0, r["acc"] - CHANCE) / (base[r["seed"]]["acc"] - CHANCE)
                    for r in rs if base[r["seed"]]["acc"] > CHANCE]
            name = k[1] + ("" if k[3] is None else " th=%g" % k[3])
            print("  M=%-3d %-14s dacc %+.4f (sd %.4f)  dbpc %+.4f (sd %.4f)  margin kept %s  n=%d" % (
                (L - 1) // 2, name, mean(da), sd(da), mean(db), sd(db),
                ("%.2f" % mean(keep)) if keep else "n/a", len(rs)))
