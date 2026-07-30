import json, glob, statistics as st
from collections import defaultdict

def ms(v):
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

def regtag(d):
    """label for the digital regularization-control arms; None = unregularized."""
    r = d.get("dig_reg") or {}
    n = r.get("dig_noise") or 0
    b = r.get("dig_bits") or 0
    w = r.get("wd") or 0
    if not (n or b or w):
        return None
    return "n%g/b%g/wd%g" % (n, b, w)


recs = defaultdict(dict)   # (variant, theta, regtag) -> seed -> record
for f in sorted(glob.glob("ssm3way_runs/*charlm*.json")):
    d = json.load(open(f))
    a = d.get("analog") or {}
    recs[(d["variant"], a.get("theta"), regtag(d))][d["seed"]] = d

dig = recs[("digital", None, None)]   # paired reference: NO regularization
print("%-11s %-6s %-14s %2s %17s %17s %9s %9s %11s %11s"
      % ("variant", "theta", "dig_reg", "n", "bpc", "acc", "emit", "rate_st",
         "paired dbpc", "pJ/tok"))
out = []
for (v, th, rt), byseed in sorted(
        recs.items(),
        key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1, x[0][2] or "")):
    seeds = sorted(byseed)
    b = ms([byseed[s]["bpc"] for s in seeds])
    ac = ms([byseed[s]["acc"] for s in seeds])
    em = ms([byseed[s]["rate_emitted"] for s in seeds])
    rs = ms([byseed[s]["rate_state"] for s in seeds])
    pj = ms([byseed[s]["energy_pJ_per_token"] for s in seeds])
    common = [s for s in seeds if s in dig]
    dl = [byseed[s]["bpc"] - dig[s]["bpc"] for s in common]
    dm, dsd = ms(dl) if dl else (float("nan"), 0.0)
    print("%-11s %-6s %-14s %2d %10.4f+-%.4f %10.4f+-%.4f %9.4f %9.4f %+7.4f+-%.4f %11.0f"
          % (v, str(th), str(rt), len(seeds), b[0], b[1], ac[0], ac[1],
             em[0], rs[0], dm, dsd, pj[0]))
    out.append((v, th, rt, len(seeds), b, em, dm, dsd, [round(x, 4) for x in dl]))

print()
print("per-seed paired dbpc vs digital (positive = worse than digital):")
for v, th, rt, n, b, em, dm, dsd, dl in out:
    if v == "digital" and rt is None:
        continue
    print("  %-11s theta=%-5s reg=%-14s n=%d  emit %.4f  dbpc %+.4f+-%.4f  per-seed %s"
          % (v, str(th), str(rt), n, em[0], dm, dsd, dl))
