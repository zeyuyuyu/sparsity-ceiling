#!/usr/bin/env python
"""3-seed confirmation aggregation for the gated (eventro/outrand) cells.
Pre-registered criteria (commit d84e8f3):
  CONFIRMED = charlm gate penalty >=0.7 bpc at every tested ot (both arms)
              AND copy margin kept <=0.15 (both L)
              AND outrand random-hold >=0.7 bpc at p=0.90 (both arms)
              AND delta-gate <= random at the low rate.
  WEAKENED  = any 3-seed mean effect < half its s0 size -> re-scope that claim.
Usage: python agg_seed12.py   (run from /work/zeyuwang/neuro_poc)
"""
import json, math, os, glob

RUNS = "ssm3way_runs"
CHANCE = 0.0625
SEEDS = [0, 1, 2]

def load(name):
    p = os.path.join(RUNS, name)
    with open(p) as f:
        return json.load(f)

def mean(xs): return sum(xs) / len(xs)
def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0

def stats(vals):
    return f"{mean(vals):.4f}±{sd(vals):.4f}"

weakened = []   # (claim, s0_effect, mean_effect)
def check_weaken(label, s0_eff, mean_eff):
    if abs(mean_eff) < 0.5 * abs(s0_eff):
        weakened.append((label, s0_eff, mean_eff))

print("=" * 78)
print("3-SEED CONFIRMATION — gated cells (pre-registration: d84e8f3)")
print("=" * 78)

# ---------- char-LM eventro ----------
print("\n[1] char-LM event-readout (theta-gate), Δbpc vs same-arm no-gate ref")
charlm_ok = True
arms = {
    "digital": ("digital_charlm_s{s}_reg_n0.02_ot{ot}.json", "digital_charlm_s{s}_reg_n0.02.json"),
    "analog":  ("analog_charlm_s{s}_theta0.15_ot{ot}.json",  "analog_charlm_s{s}_theta0.15.json"),
}
for arm, (gpat, rpat) in arms.items():
    refs = {s: load(rpat.format(s=s))["bpc"] for s in SEEDS}
    print(f"  {arm}: no-gate ref bpc {stats([refs[s] for s in SEEDS])}")
    for ot in ["0.02", "0.1", "1.0"]:
        cells = {s: load(gpat.format(s=s, ot=ot)) for s in SEEDS}
        dl = [cells[s]["bpc"] - refs[s] for s in SEEDS]
        ro = [cells[s]["rate_out"] for s in SEEDS]
        m = mean(dl)
        flag = "PASS(>=0.7)" if m >= 0.7 else "FAIL(<0.7)"
        if m < 0.7: charlm_ok = False
        print(f"    ot={ot:>4}: bpc {stats([cells[s]['bpc'] for s in SEEDS])}  "
              f"Δbpc {stats(dl)}  r_out {stats(ro)}  {flag}")
        check_weaken(f"charlm {arm} ot={ot} Δbpc", dl[0], m)

# ---------- copy eventro ----------
print("\n[2] copy event-readout: margin kept vs 3-seed no-gate ref (chance 0.0625)")
copy_ok = True
for L in [33, 65]:
    ref_acc = [load(f"digital_copy_L{L}_s{s}_reg_n0.02_ep30.json")["acc"] for s in SEEDS]
    ref_margin = mean(ref_acc) - CHANCE
    print(f"  L={L} (M={(L-1)//2}): ref acc {stats(ref_acc)}  margin {ref_margin:.4f}")
    for ot in ["0.02", "0.1", "1.0"]:
        cells = {s: load(f"digital_copy_L{L}_s{s}_reg_n0.02_ot{ot}_ep30.json") for s in SEEDS}
        acc = [cells[s]["acc"] for s in SEEDS]
        mk = [(a - CHANCE) / ref_margin for a in acc]
        ro = [cells[s]["rate_out"] for s in SEEDS]
        m = mean(mk)
        flag = "PASS(<=0.15)" if m <= 0.15 else "FAIL(>0.15)"
        if m > 0.15: copy_ok = False
        print(f"    ot={ot:>4}: acc {stats(acc)}  margin-kept {stats(mk)}  r_out {stats(ro)}  {flag}")
        # weaken check on the COLLAPSE effect = margin destroyed (1 - mk)
        check_weaken(f"copy L={L} ot={ot} margin-destroyed", 1 - mk[0], 1 - m)

# ---------- outrand ----------
print("\n[3] outrand (random-hold) vs delta-gate, Δbpc vs same-arm no-gate ref")
outrand_ok = True
rand_cells = {
    ("digital", "0.90"): "digital_charlm_s{s}_reg_n0.02_pr0.90.json",
    ("digital", "0.08"): "digital_charlm_s{s}_reg_n0.02_pr0.08.json",
    ("analog",  "0.90"): "analog_charlm_s{s}_theta0.15_pr0.90.json",
}
rand_d = {}
for (arm, p), pat in rand_cells.items():
    rpat = arms[arm][1]
    refs = {s: load(rpat.format(s=s))["bpc"] for s in SEEDS}
    cells = {s: load(pat.format(s=s)) for s in SEEDS}
    dl = [cells[s]["bpc"] - refs[s] for s in SEEDS]
    ro = [cells[s]["rate_out"] for s in SEEDS]
    rand_d[(arm, p)] = mean(dl)
    print(f"  {arm} p={p}: bpc {stats([cells[s]['bpc'] for s in SEEDS])}  "
          f"Δbpc {stats(dl)}  r_out {stats(ro)}")
    check_weaken(f"outrand {arm} p={p} Δbpc", dl[0], mean(dl))
for arm in ["digital", "analog"]:
    if rand_d[(arm, "0.90")] < 0.7:
        outrand_ok = False
        print(f"  !! {arm} random-hold at p=0.90 < 0.7 bpc")
# low-rate delta vs random: digital ot1.0 (r_out~0.08) vs pr0.08
d_delta = mean([load(f"digital_charlm_s{s}_reg_n0.02_ot1.0.json")["bpc"]
                - load(f"digital_charlm_s{s}_reg_n0.02.json")["bpc"] for s in SEEDS])
d_rand = rand_d[("digital", "0.08")]
print(f"  low-rate check (digital, r_out≈0.08): delta-gate Δbpc {d_delta:.4f} "
      f"vs random Δbpc {d_rand:.4f} -> delta {'<=' if d_delta <= d_rand else '>'} random")
if d_delta > d_rand: outrand_ok = False

# ---------- verdict ----------
print("\n" + "=" * 78)
print(f"criterion charlm >=0.7 both arms all ot : {'MET' if charlm_ok else 'NOT MET'}")
print(f"criterion copy margin<=0.15 both L      : {'MET' if copy_ok else 'NOT MET'}")
print(f"criterion outrand (>=0.7 @p0.90 + delta<=random low-rate): {'MET' if outrand_ok else 'NOT MET'}")
if weakened:
    print("\nWEAKENED claims (3-seed mean < half of s0 effect):")
    for label, s0e, me in weakened:
        print(f"  - {label}: s0 {s0e:.4f} -> mean {me:.4f}")
else:
    print("\nNo claim weakened (every 3-seed mean effect >= half its s0 size).")
verdict = "CONFIRMED" if (charlm_ok and copy_ok and outrand_ok and not weakened) else \
          ("CONFIRMED-WITH-RESCOPE" if (charlm_ok and copy_ok and outrand_ok) else "NOT CONFIRMED")
print(f"\nPRE-REGISTERED VERDICT: {verdict}")
