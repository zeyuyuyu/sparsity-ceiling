"""
Is the stranded-readout verdict a property of the ANALOG ROUTE, or a property
of char-LM's MODEL SHAPE?

BACKGROUND.  energy_datapath.py established, on char-LM (V=284, E=64, H=256),
that W_out carries ~75% of the analog SSM's pJ/token while the recurrence W_mix
-- the only term any neuromorphic mechanism in this literature touches -- carries
~8%, and that even a FREE recurrence leaves analog at only 1.74x vs the
regularized digital reference.  The event-readout experiments then showed the
readout cannot be gated without ~+1.0 bpc, so that 75% is stranded.

That verdict was stated as a general one.  But H*V > H*H holds only because
char-LM's output vocabulary is WIDER than its state (V/H = 1.11).  A streaming
classification workload -- the exact workload class this project already
recommends the analog route for -- has V/H ~ 0.04, where the recurrence is the
largest matrix in the model and the readout is negligible.  This script asks
what the SAME measured event activity is worth at that shape.

It re-uses energy_datapath.terms() verbatim, so nothing here re-derives or
re-prices anything; the only new thing is sweeping the shape (V, H) and the
input event rate.

Also fixes an analysis bug in energy_datapath.py's label(): it ignores
out_theta / out_prand, so with the readout-gated cells now on disk it POOLS
gated and no-gate cells into one group (analog th=0.15 showed n=20 and a bpc of
4.018, which is the GATED collapse value, not the no-gate 3.185).  Every cell
here is filtered to the no-gate datapath.

Usage:  python energy_shape.py [runs_dir]
"""
import glob
import json
import os
import sys

from energy_datapath import (E_AC, E_MAC, E_CONV_DEFAULT, E_CONV_SWEEP,
                             E_STORE_DEFAULT, E_STORE_SWEEP, terms)

# shapes of interest.  V is the readout width (output classes / vocabulary).
SHAPES = [
    (10,   "10-class streaming (FashionMNIST rows, DVS gesture, keyword spot)"),
    (20,   "20-class streaming"),
    (35,   "spoken-digit / small-keyword vocabulary"),
    (65,   "char-LM, minimal alphabet (paper 1's charlm vocab)"),
    (284,  "char-LM as measured here  <-- the published operating point"),
    (1024, "sub-word LM"),
    (32000, "word-piece LM (Llama-class vocabulary)"),
]


def gated(d):
    """True if this cell has the send-on-delta / random readout gate enabled."""
    return bool(d.get("out_theta") or d.get("out_prand"))


def mean_sd(v):
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5


def load_nogate(runs):
    """char-LM cells with the readout gate OFF, grouped by operating point."""
    groups = {}
    for f in sorted(glob.glob(os.path.join(runs, "*charlm*.json"))):
        d = json.load(open(f))
        if d.get("task") != "charlm" or gated(d):
            continue
        reg = d.get("dig_reg") or {}
        if d["variant"] == "analog":
            lab = f"analog th={d['analog']['theta']}"
        elif d["variant"] == "digital" and (reg.get("dig_noise") or
                                            reg.get("dig_bits") or reg.get("wd")):
            n = f"n{reg['dig_noise']}" if reg.get("dig_noise") else ""
            b = f"+b{reg['dig_bits']}" if reg.get("dig_bits") else ""
            w = f" wd={reg['wd']}" if reg.get("wd") else ""
            lab = f"digital+{n}{b}{w}".strip()
        else:
            lab = d["variant"]
        groups.setdefault(lab, []).append(d)
    return groups


def total(variant, V, E, H, r_z, r_s, datapath, r_in=None,
          e_conv=E_CONV_DEFAULT, e_store=E_STORE_DEFAULT):
    return sum(terms(variant, V, E, H, r_z, r_s, datapath,
                     e_conv=e_conv, e_store=e_store, r_in=r_in).values())


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else "ssm3way_runs"
    groups = load_nogate(runs)
    if not groups:
        print("no un-gated char-LM cells found in", runs)
        return

    print("=" * 78)
    print("PART 0 -- CORRECTED no-gate decomposition")
    print("(energy_datapath.py's label() pools readout-gated cells; filtered out")
    print(" here.  Compare n and bpc against that script's PART 1.)")
    print("=" * 78)
    print(f"{'cell':22s} {'n':>2s} {'bpc':>7s} {'r_z':>6s} {'r_s':>6s} "
          f"{'pJ/tok':>9s} {'W_in%':>6s} {'W_mix%':>7s} {'W_out%':>7s}")
    for lab in sorted(groups):
        ds = groups[lab]
        d0 = ds[0]
        bpc, bsd = mean_sd([d["bpc"] for d in ds])
        rz, _ = mean_sd([d["rate_emitted"] for d in ds])
        rs, _ = mean_sd([d["rate_state"] for d in ds])
        t = terms(d0["variant"], d0["vocab"], d0["E"], d0["H"], rz, rs,
                  "current", e_conv=0.0, e_store=0.0)   # published proxy
        tot = sum(t.values())
        print(f"{lab:22s} {len(ds):2d} {bpc:7.4f} {rz:6.3f} {rs:6.3f} {tot:9.0f} "
              f"{100*t['W_in']/tot:6.1f} {100*t['W_mix']/tot:7.1f} "
              f"{100*t['W_out']/tot:7.1f}")

    # operating point: the published analog cell and the regularized digital ref
    A_LAB = "analog th=0.15"
    R_LAB = "digital+n0.02"
    da, dr = groups[A_LAB], groups[R_LAB]
    d0 = da[0]
    E, H = d0["E"], d0["H"]
    V_MEAS = d0["vocab"]
    rz_a, rz_sd = mean_sd([d["rate_emitted"] for d in da])
    rs_a, _ = mean_sd([d["rate_state"] for d in da])
    bpc_a, _ = mean_sd([d["bpc"] for d in da])
    bpc_r, _ = mean_sd([d["bpc"] for d in dr])
    print(f"\noperating point held FIXED for every sweep below:")
    print(f"  analog  = {A_LAB}: r_z = {rz_a:.4f} +/- {rz_sd:.4f} (n={len(da)}), "
          f"r_state = {rs_a:.4f}, bpc = {bpc_a:.4f}")
    print(f"  ref     = {R_LAB}: bpc = {bpc_r:.4f} (n={len(dr)}), r_z = 1.0")
    print(f"  quality cost of the analog datapath = "
          f"{bpc_a - bpc_r:+.4f} bpc  (char-LM, 3 seeds)")
    print(f"  E = {E}, H = {H}; only V is varied (and H in PART 2).")
    print("  NOTE r_z and the quality cost are char-LM MEASUREMENTS.  Every ratio")
    print("  below is a PROJECTION of that activity onto another model shape; no")
    print("  quality number at any other shape has been measured.")

    print()
    print("=" * 78)
    print("PART 1 -- THE SHAPE LAW: analog's pJ advantage vs readout width V")
    print("H, E and the measured r_z held fixed; only the output width changes.")
    print("=" * 78)
    print(f"{'V':>6s} {'V/H':>6s} {'W_out% (ana)':>13s} {'W_mix% (ana)':>13s} "
          f"{'ref pJ/tok':>11s} {'ana pJ/tok':>11s} {'ratio':>7s} {'ceil':>7s}")
    for V, note in SHAPES:
        t = terms("analog", V, E, H, rz_a, rs_a, "current")
        ta = sum(t.values())
        tr = total("digital", V, E, H, 1.0, 1.0, "current")
        # ceiling = analog with a FREE recurrence at this shape
        tc = total("analog", V, E, H, 0.0, rs_a, "current")
        print(f"{V:6d} {V/H:6.2f} {100*t['W_out']/ta:13.1f} "
              f"{100*t['W_mix']/ta:13.1f} {tr:11.0f} {ta:11.0f} "
              f"{tr/ta:6.2f}x {tr/tc:6.2f}x")
    print()
    print("READING: the ratio column is what the SAME measured event activity is")
    print("worth at each shape; the ceil column is the most a perfectly free")
    print("recurrence could ever deliver there.  Where V >~ H the two collapse")
    print("together and both are small -- that is the stranded readout.  Where")
    print("V << H the recurrence is the dominant matrix and the measured activity")
    print("converts to a much larger cut.")

    print()
    print("=" * 78)
    print("PART 2 -- the same law in the other variable: state width H at V=10")
    print("=" * 78)
    print(f"{'H':>6s} {'V/H':>6s} {'ref pJ/tok':>11s} {'ana pJ/tok':>11s} "
          f"{'ratio':>7s}")
    for Hx in (64, 128, 256, 512, 1024):
        ta = total("analog", 10, E, Hx, rz_a, rs_a, "current")
        tr = total("digital", 10, E, Hx, 1.0, 1.0, "current")
        print(f"{Hx:6d} {10/Hx:6.2f} {tr:11.0f} {ta:11.0f} {tr/ta:6.2f}x")
    print("READING: the advantage GROWS with H at fixed V, because the term the")
    print("analog datapath event-prices is the one that scales as H^2.  This is")
    print("the opposite of the char-LM regime and it is a testable prediction.")

    print()
    print("=" * 78)
    print("PART 3 -- goal item (1): is a SPARSE EVENT INPUT worth anything?")
    print("Priced SYMMETRICALLY: a genuinely event-like input stream is available")
    print("to the digital SSM too, so BOTH arms get W_in at r_in (this is the same")
    print("lesson the readout row taught -- digital got the same gate for free).")
    print("=" * 78)
    print(f"{'V':>6s} {'r_in':>6s} {'ref pJ/tok':>11s} {'ana pJ/tok':>11s} "
          f"{'ratio':>7s} {'vs dense input':>15s}")
    for V in (10, 284):
        base = None
        for r_in in (1.0, 0.5, 0.2, 0.05):
            dp = "current" if r_in >= 1.0 else "event_input"
            ta = total("analog", V, E, H, rz_a, rs_a, dp, r_in=r_in)
            tr = total("digital", V, E, H, 1.0, 1.0, dp, r_in=r_in)
            if base is None:
                base = tr / ta
            print(f"{V:6d} {r_in:6.2f} {tr:11.0f} {ta:11.0f} {tr/ta:6.2f}x "
                  f"{tr/ta - base:+14.2f}")
    print()
    print("READING: compare the spread WITHIN a V block (what input sparsity buys)")
    print("against the gap BETWEEN the two V blocks (what output width costs).")
    print("If the between-V gap dominates, then goal item (1) was correctly")
    print("deferred -- but for the wrong reason: not because W_in's share is small")
    print("at char-LM's shape, but because input pricing is second-order at EVERY")
    print("shape, while the output width is the whole effect.")

    print()
    print("=" * 78)
    print("PART 4 -- do the converter/storage costs bite once the matrices ARE")
    print("cheap?  (energy_datapath PART 4 found 0.00-0.33% at V=284; the note")
    print("there was that they 'only start to matter once the matrices are cheap',")
    print("which is exactly the V=10 regime.  Checking that, not assuming it.)")
    print("=" * 78)
    print(f"{'V':>6s} {'E_conv':>7s} {'E_store':>8s} {'ana pJ/tok':>11s} "
          f"{'conv+store %':>13s} {'ratio':>7s}")
    for V in (10, 284):
        for ec in E_CONV_SWEEP:
            for es in E_STORE_SWEEP:
                t = terms("analog", V, E, H, rz_a, rs_a, "current",
                          e_conv=ec, e_store=es)
                ta = sum(t.values())
                tr = total("digital", V, E, H, 1.0, 1.0, "current")
                extra = t["storage"] + t["converters"]
                print(f"{V:6d} {ec:7.2f} {es:8.2f} {ta:11.0f} "
                      f"{100*extra/ta:12.2f}% {tr/ta:6.2f}x")

    print()
    print("The sweep above uses the CURRENT datapath, where W_in is still a dense")
    print("MAC and therefore still dominates at V=10.  The corner where the")
    print("converters have the best chance of mattering is the CHEAPEST datapath")
    print("(V=10 AND event input), so check that explicitly rather than inferring:")
    print(f"{'V':>6s} {'r_in':>6s} {'E_conv':>7s} {'E_store':>8s} "
          f"{'ana pJ/tok':>11s} {'conv+store %':>13s}")
    for ec, es in ((0.06, 0.02), (1.0, 0.5)):
        t = terms("analog", 10, E, H, rz_a, rs_a, "event_input", r_in=0.05,
                  e_conv=ec, e_store=es)
        ta = sum(t.values())
        extra = t["storage"] + t["converters"]
        print(f"{10:6d} {0.05:6.2f} {ec:7.2f} {es:8.2f} {ta:11.0f} "
              f"{100*extra/ta:12.2f}%")

    print()
    print("=" * 78)
    print("PART 5 -- what is and is not established")
    print("=" * 78)
    print("ESTABLISHED (arithmetic over the published accounting, no new run):")
    print("  * the 75%-readout / 8%-recurrence split and the 1.74x free-recurrence")
    print("    ceiling are properties of char-LM's SHAPE (V/H = 1.11), not of the")
    print("    analog route.  Restating them as general claims would be wrong.")
    print("NOT ESTABLISHED (needs a run; this is the pre-registration target):")
    print("  * that the measured event activity r_z ~ 0.64 TRANSFERS to a small-V")
    print("    streaming task.  Activity is task-dependent -- theta already failed")
    print("    to transfer from char-LM to copy.")
    print("  * that the analog state's quality cost stays ~+0.16 bpc there.  On")
    print("    copy it was +0.9 bpc.  A streaming classification task is")
    print("    statistical, not precise-recall, so the datapath-degradation")
    print("    principle predicts cheap -- but that principle has already made")
    print("    one wrong prediction (the event readout on copy), so it is a")
    print("    hypothesis, not a result.")


if __name__ == "__main__":
    main()
