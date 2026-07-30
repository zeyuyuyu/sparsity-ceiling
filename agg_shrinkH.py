import json, glob, math, statistics as st
from collections import defaultdict

DEMAND = 16 * math.log2(16)   # M*log2(K) = 64 bits
CHANCE = 0.0625

def Hb(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def floor_for(H):
    t = DEMAND / H
    if t >= 1:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if Hb(mid) < t:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def ms(v):
    v = [x for x in v if x == x]
    if not v:
        return (float('nan'), 0.0)
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)

rows = defaultdict(list)
for f in sorted(glob.glob("ssm3way_runs/*_H*_ep30.json")):
    d = json.load(open(f))
    d['_H'] = int(f.split('_H')[1].split('_')[0])
    d['_seed'] = int(f.split('_s')[1].split('_')[0])
    rows[(d['variant'], d['_H'])].append(d)

sample = next(iter(rows.values()))[0]
print("record keys:", sorted(sample.keys()))
print()

def get(d, *names):
    for n in names:
        if n in d:
            return d[n]
    return float('nan')

hdr = ("variant", "H", "n", "acc", "bpc", "rate_state", "emit", "pJ/tok", "floor", "H*Hb(rho)")
print("%-11s %4s %2s %17s %17s %17s %7s %10s %7s %9s" % hdr)
dig = {}
per = {}
for (v, H), ds in sorted(rows.items(), key=lambda x: (x[0][0], -x[0][1])):
    a = ms([d['acc'] for d in ds])
    b = ms([d['bpc'] for d in ds])
    rs = ms([get(d, 'rate_state', 'rate_hidden') for d in ds])
    em = ms([get(d, 'rate_emitted', 'rate_out', 'rate_output') for d in ds])
    pj = ms([get(d, 'pJ_per_token', 'energy_pJ_per_token', 'pJ_tok') for d in ds])
    cert = H * Hb(rs[0]) if rs[0] == rs[0] else float('nan')
    if v == 'digital':
        dig[H] = a[0]
    per[(v, H)] = dict(acc=a, bpc=b, rs=rs, em=em, pj=pj, cert=cert, n=len(ds),
                       accs={d['_seed']: d['acc'] for d in ds},
                       bpcs={d['_seed']: d['bpc'] for d in ds})
    print("%-11s %4d %2d %10.4f+-%.4f %10.4f+-%.4f %10.4f+-%.4f %7.3f %10.0f %7.3f %9.1f"
          % (v, H, len(ds), a[0], a[1], b[0], b[1], rs[0], rs[1], em[0], pj[0], floor_for(H), cert))

print()
for (v, H), r in sorted(per.items(), key=lambda x: (x[0][0], -x[0][1])):
    if v == 'digital':
        continue
    mk = (r['acc'][0] - CHANCE) / (dig[H] - CHANCE)
    dref = per[('digital', H)]
    common = sorted(set(r['bpcs']) & set(dref['bpcs']))
    dl = [r['bpcs'][s] - dref['bpcs'][s] for s in common]
    dm, dsd = ms(dl)
    print("margin_kept %-11s H=%3d : %.3f   paired dbpc %+.4f+-%.4f (n=%d)"
          % (v, H, mk, dm, dsd, len(common)))
