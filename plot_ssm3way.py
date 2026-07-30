"""Figure for the 3-way (4-variant) SSM comparison: workload dichotomy + bound test.

Panel A -- quality retained vs memory load M on the copy task.  "Margin kept" is
the fraction of the digital baseline's above-chance accuracy margin a variant
retains; with absolute accuracies this low, raw accuracy is misleading and the
margin is the honest normalisation.

Panel B -- recurrent-state activity vs M for the spiking-state variant, against
the firing-floor bound.  The bound is rho >= H_b^-1(log2 M_states / H), where
M_states is the number of DISTINGUISHABLE memory states, not the symbol count.
Copying M symbols from a K-letter alphabet requires distinguishing K^M states,
so log2 M_states = M * log2 K.  At H=256, K=16 that is 64 / 128 / 256 bits for
M = 16 / 32 / 64, i.e. predicted floors 0.042 / 0.110 / 0.500.  The bound only
binds on a net that actually retains the sequence -- a net at chance stores
nothing and is trivially consistent with any floor -- so the accuracy panel has
to be read alongside it.

Reads logged JSON only; no training.  Incomplete cells are plotted with the
seed count annotated rather than silently dropped or silently averaged.
Usage: python plot_ssm3way.py [epochs]   (default 30)
"""
import json, glob, os, sys, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/work/zeyuwang/neuro_poc/ssm3way_runs"
OUT = "/work/zeyuwang/neuro_poc/fig_ssm3way.pdf"
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K = 16            # copy alphabet
CHANCE = 1.0 / K
H = 256           # hidden width used by every variant
THETA_MAIN = 0.2  # the calibrated analog threshold for the copy row

VARIANTS = [("digital", "digital SSM", "#444444", "o"),
            ("spikeout", "continuous state + spiking output", "#1f77b4", "s"),
            ("analog", "analog state + events", "#d62728", "^"),
            ("spikestate", "spiking state (floor control)", "#2ca02c", "v")]


def mean(x):
    return sum(x) / len(x)


def sd(x):
    if len(x) < 2:
        return 0.0
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def h_binary(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def h_binary_inv(y):
    """Smallest p in [0, 0.5] with H_b(p) >= y; 0.5 if y > 1."""
    if y <= 0:
        return 0.0
    if y >= 1:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if h_binary(mid) < y:
            lo = mid
        else:
            hi = mid
    return hi


def load():
    """(variant, M) -> {seed: record}, de-duplicated by seed."""
    cells = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        r = json.load(open(f))
        if r.get("task") != "copy" or r["epochs"] != EPOCHS:
            continue
        th = r["analog"]["theta"] if r.get("analog") else None
        if r["variant"] == "analog" and th != THETA_MAIN:
            continue          # calibration sweep cells are not row cells
        cells[(r["variant"], (r["L"] - 1) // 2)].setdefault(r["seed"], r)
    return cells


cells = load()
Ms = sorted({m for _, m in cells})
if not Ms:
    sys.exit("no copy cells at %d epochs" % EPOCHS)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 3.9))

# ---- Panel A: margin kept vs M -------------------------------------------
for name, label, colour, marker in VARIANTS:
    if name == "digital":
        continue
    xs, ys, es, ns = [], [], [], []
    for M in Ms:
        base = cells.get(("digital", M))
        cell = cells.get((name, M))
        if not base or not cell:
            continue
        dbase = mean([r["acc"] for r in base.values()]) - CHANCE
        if dbase <= 0:
            continue
        margins = [(r["acc"] - CHANCE) / dbase for r in cell.values()]
        xs.append(M)
        ys.append(mean(margins))
        es.append(sd(margins))
        ns.append(len(margins))
    if not xs:
        continue
    axA.errorbar(xs, ys, yerr=es, color=colour, marker=marker, capsize=3, label=label)
    for x, y, n in zip(xs, ys, ns):
        if n < 3:
            axA.annotate("n=%d" % n, (x, y), textcoords="offset points",
                         xytext=(4, 5), fontsize=7, color=colour)

axA.axhline(1.0, color="#444444", ls="--", lw=1)
axA.text(Ms[0], 1.02, "digital baseline", fontsize=7, color="#444444")
axA.axhline(0.0, color="grey", ls=":", lw=1)
axA.set_xscale("log", base=2)
axA.set_xticks(Ms)
axA.set_xticklabels([str(m) for m in Ms])
axA.set_xlabel("memory load $M$ (symbols to recall)")
axA.set_ylabel("above-chance accuracy margin kept")
axA.set_title("(a) quality retained on copy", fontsize=10)
axA.legend(fontsize=7, loc="best")

# ---- Panel B: state activity vs the firing-floor bound --------------------
bound = [h_binary_inv(M * math.log2(K) / H) for M in Ms]
axB.plot(Ms, bound, color="black", ls="--", lw=1.4,
         label=r"bound $H_b^{-1}(\log_2 M_{\rm states}/H)$")
axB.fill_between(Ms, 0, bound, color="black", alpha=0.06)

for name, label, colour, marker in VARIANTS:
    if name == "digital":
        continue
    key = "rate_state" if name == "spikestate" else "rate_emitted"
    xs, ys, es = [], [], []
    for M in Ms:
        cell = cells.get((name, M))
        if not cell:
            continue
        vals = [r[key] for r in cell.values()]
        xs.append(M)
        ys.append(mean(vals))
        es.append(sd(vals))
    if xs:
        axB.errorbar(xs, ys, yerr=es, color=colour, marker=marker, capsize=3,
                     label="%s (%s)" % (label.split(" (")[0], key))

axB.set_xscale("log", base=2)
axB.set_xticks(Ms)
axB.set_xticklabels([str(m) for m in Ms])
axB.set_ylim(0, 1.05)
axB.set_xlabel("memory load $M$ (symbols to recall)")
axB.set_ylabel("activity (fraction of units active)")
axB.set_title("(b) activity vs the firing floor", fontsize=10)
axB.legend(fontsize=7, loc="best")

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)

# ---- console companion: the numbers behind the figure ---------------------
print("\nbound at H=%d, K=%d (log2 M_states = M*log2 K):" % (H, K))
for M, b in zip(Ms, bound):
    print("  M=%-3d  %3d bits / %d units -> floor rho >= %.4f" %
          (M, M * int(math.log2(K)), H, b))
print("\ncell coverage (seeds present):")
for name, _, _, _ in VARIANTS:
    row = []
    for M in Ms:
        c = cells.get((name, M))
        row.append("M=%d:%s" % (M, ",".join(str(s) for s in sorted(c)) if c else "-"))
    print("  %-11s %s" % (name, "  ".join(row)))
