"""
Energy-datapath accounting for the 4-way SSM comparison.

WHY THIS EXISTS.  The published pJ/token proxy in ssm3way.py prices ONLY the
recurrence (W_mix) as event-driven; the input projection W_in and the readout
W_out are charged as dense MACs for digital/spikeout/analog.  Consequence: the
measured ~27-62% event activity of the analog-state SSM is an ACTIVITY win but
NOT a pJ win, and the proxy ranks spikestate (which gets both W_mix AND W_out
event-priced) cheapest despite it being the worst variant on quality.

This script does three things the proxy does not:
  (1) DECOMPOSE  -- per-term share of pJ/token for every measured char-LM cell,
      so "which layer blocks the win" is a number, not an intuition.
  (2) PROJECT    -- re-price W_in and/or W_out as event-driven and report the
      resulting pJ/token.  Labelled PROJECTION: the input/output event rates are
      ASSUMED here (set equal to the measured emitted rate), not measured.  The
      event-readout row of ssm3way.py measures r_out for real.
  (3) PRICE THE UNPRICED -- the analog storage element and the ADC/DAC
      conversion per graded event, which the proxy charges nothing for, with a
      sensitivity sweep so the verdict does not rest on a guessed constant.

Nothing here re-derives a quality number; it reads them from the run JSONs.

Usage:  python energy_datapath.py [runs_dir]
"""
import glob
import json
import os
import sys

# 45 nm Horowitz, same constants as ssm3way.py (pJ)
E_MAC, E_AC = 4.6, 0.9

# --- costs the published proxy charges nothing for -------------------------
# Converter: one conversion per graded analog event leaving/entering the analog
# domain.  A 6-bit conversion at a mid-range SAR FoM of ~10 fJ per
# conversion-step is ~0.06 pJ.  Older/relaxed designs are far worse, so the
# verdict is swept over four decades rather than asserted at one value.
E_CONV_DEFAULT = 0.06
E_CONV_SWEEP = (0.01, 0.06, 0.25, 1.0)
# Analog storage element: leak/refresh per state unit per timestep.  No solid
# public number at this node for an SSM-style capacitor/memristor state, so this
# is an explicit ASSUMPTION, swept.
E_STORE_DEFAULT = 0.02
E_STORE_SWEEP = (0.0, 0.02, 0.1, 0.5)

DATAPATHS = ("current", "event_readout", "event_input", "full_event")


def terms(variant, V, E, H, r_z, r_s, datapath,
          e_conv=E_CONV_DEFAULT, e_store=E_STORE_DEFAULT, r_in=None, r_out=None):
    """pJ/token per term.  r_in/r_out default to the measured emitted rate."""
    win, wmix, wout = E * H, H * H, H * V
    if r_in is None:
        r_in = r_z
    if r_out is None:
        r_out = r_z
    ev_in = datapath in ("event_input", "full_event")
    ev_out = datapath in ("event_readout", "full_event")
    t = {}

    # ---- input projection
    if ev_in:
        t["W_in"] = win * r_in * E_AC
    else:
        t["W_in"] = win * E_MAC

    # ---- state / recurrence (as published in ssm3way.py)
    if variant == "digital":
        t["state"] = H * E_MAC
        t["W_mix"] = wmix * E_MAC
    elif variant == "spikeout":
        t["state"] = H * E_MAC
        t["W_mix"] = wmix * r_z * E_AC
    elif variant == "analog":
        t["state"] = H * E_AC
        t["W_mix"] = wmix * r_z * E_AC
    else:                                            # spikestate
        t["state"] = H * E_MAC
        t["W_mix"] = wmix * r_s * E_AC

    # ---- readout
    rate_ro = r_s if variant == "spikestate" else r_out
    if ev_out or variant == "spikestate":
        t["W_out"] = wout * rate_ro * E_AC
    else:
        t["W_out"] = wout * E_MAC

    # ---- costs the proxy omits.  Only the analog datapath crosses the
    # analog/digital boundary, so only it pays converters and storage.
    if variant == "analog":
        t["storage"] = H * e_store
        t["converters"] = H * r_z * e_conv
    else:
        t["storage"] = 0.0
        t["converters"] = 0.0
    return t


def load(runs):
    cells = {}
    for f in sorted(glob.glob(os.path.join(runs, "*charlm*.json"))):
        d = json.load(open(f))
        if d.get("task") != "charlm":
            continue
        cells[os.path.basename(f)[:-5]] = d
    return cells


def label(name, d):
    if d["variant"] == "analog":
        return f"analog th={d['analog']['theta']}"
    reg = d.get("dig_reg") or {}
    if d["variant"] == "digital" and (reg.get("dig_noise") or reg.get("dig_bits")
                                      or reg.get("wd")):
        bits = f"+b{reg['dig_bits']}" if reg.get("dig_bits") else ""
        wd = f" wd={reg['wd']}" if reg.get("wd") else ""
        n = f"n{reg['dig_noise']}" if reg.get("dig_noise") else ""
        return f"digital+{n}{bits}{wd}".strip()
    return d["variant"]


def mean_sd(v):
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else "ssm3way_runs"
    cells = load(runs)
    if not cells:
        print("no char-LM cells found in", runs)
        return

    # group seeds
    groups = {}
    for name, d in cells.items():
        groups.setdefault(label(name, d), []).append(d)

    print("=" * 78)
    print("PART 1 -- DECOMPOSITION of the PUBLISHED proxy (datapath 'current')")
    print("Which layer actually carries the pJ?  shares of the total.")
    print("=" * 78)
    print(f"{'cell':22s} {'n':>2s} {'bpc':>7s} {'r_z':>6s} "
          f"{'pJ/tok':>9s} {'W_in%':>6s} {'W_mix%':>7s} {'W_out%':>7s}")
    for lab in sorted(groups):
        ds = groups[lab]
        d0 = ds[0]
        bpc, _ = mean_sd([d["bpc"] for d in ds])
        rz, _ = mean_sd([d["rate_emitted"] for d in ds])
        rs, _ = mean_sd([d["rate_state"] for d in ds])
        t = terms(d0["variant"], d0["vocab"], d0["E"], d0["H"], rz, rs, "current",
                  e_conv=0.0, e_store=0.0)          # published proxy: unpriced
        tot = sum(t.values())
        print(f"{lab:22s} {len(ds):2d} {bpc:7.4f} {rz:6.3f} {tot:9.0f} "
              f"{100*t['W_in']/tot:6.1f} {100*t['W_mix']/tot:7.1f} "
              f"{100*t['W_out']/tot:7.1f}")

    # the reference cell for ratios: the properly regularized digital baseline
    ref_lab = "digital+n0.02"
    ref = groups.get(ref_lab) or groups["digital"]
    ref_lab = ref_lab if ref_lab in groups else "digital"
    rd = ref[0]
    V, E, H = rd["vocab"], rd["E"], rd["H"]
    rz_ref, _ = mean_sd([d["rate_emitted"] for d in ref])
    rs_ref, _ = mean_sd([d["rate_state"] for d in ref])
    ref_tot = sum(terms("digital", V, E, H, rz_ref, rs_ref, "current",
                        e_conv=0.0, e_store=0.0).values())
    print(f"\nreference for all ratios below: {ref_lab} at {ref_tot:.0f} pJ/token "
          f"(digital pays MACs everywhere in every datapath, so its total is "
          f"datapath-invariant)")

    print()
    print("=" * 78)
    print("PART 2 -- CEILING under the CURRENT datapath")
    print("What is the BEST analog could do if the recurrence cost went to ZERO?")
    print("=" * 78)
    floor = sum(terms("analog", V, E, H, 0.0, 1.0, "current").values())
    print(f"analog with r_z -> 0 (free recurrence): {floor:.0f} pJ/token "
          f"= {ref_tot/floor:.2f}x vs {ref_lab}")
    for lab in sorted(l for l in groups if l.startswith("analog")):
        ds = groups[lab]
        rz, _ = mean_sd([d["rate_emitted"] for d in ds])
        rs, _ = mean_sd([d["rate_state"] for d in ds])
        tot = sum(terms("analog", V, E, H, rz, rs, "current").values())
        print(f"  {lab:20s} r_z={rz:5.3f}  {tot:8.0f} pJ  "
              f"{ref_tot/tot:5.2f}x vs ref")
    print("READING: the ceiling is set by the two MAC-priced dense layers, not by")
    print("the event mechanism.  Sparsifying the recurrence further cannot reach it.")

    print()
    print("=" * 78)
    print("PART 3 -- PROJECTION: re-price W_in and/or W_out as event-driven")
    print("ASSUMPTION (not measurement): r_in = r_out = the cell's measured r_z.")
    print("=" * 78)
    hdr = f"{'cell':22s} {'r_z':>6s}"
    for dp in DATAPATHS:
        hdr += f" {dp[:12]:>13s}"
    print(hdr)
    for lab in sorted(groups):
        ds = groups[lab]
        d0 = ds[0]
        rz, _ = mean_sd([d["rate_emitted"] for d in ds])
        rs, _ = mean_sd([d["rate_state"] for d in ds])
        line = f"{lab:22s} {rz:6.3f}"
        for dp in DATAPATHS:
            tot = sum(terms(d0["variant"], V, E, H, rz, rs, dp).values())
            line += f" {tot:8.0f}({ref_tot/tot:4.1f}x)"
        print(line)
    print("NOTE spikestate's W_out is event-priced in EVERY column -- that is how")
    print("the published proxy already treats it, and it is why it looked cheapest.")

    print()
    print("=" * 78)
    print("PART 4 -- SENSITIVITY: do the UNPRICED analog costs change the verdict?")
    print("=" * 78)
    lab_a = "analog th=0.15" if "analog th=0.15" in groups else \
        sorted(l for l in groups if l.startswith("analog"))[0]
    da = groups[lab_a]
    rz_a, _ = mean_sd([d["rate_emitted"] for d in da])
    rs_a, _ = mean_sd([d["rate_state"] for d in da])
    print(f"cell = {lab_a}, r_z = {rz_a:.3f}, H = {H}")
    print(f"{'E_conv (pJ/ev)':>15s} {'E_store (pJ/unit/step)':>23s} "
          f"{'current':>10s} {'full_event':>12s} {'conv+store %':>13s}")
    for ec in E_CONV_SWEEP:
        for es in E_STORE_SWEEP:
            tc = terms("analog", V, E, H, rz_a, rs_a, "current", ec, es)
            tf = terms("analog", V, E, H, rz_a, rs_a, "full_event", ec, es)
            extra = tf["storage"] + tf["converters"]
            print(f"{ec:15.2f} {es:23.2f} {sum(tc.values()):10.0f} "
                  f"{sum(tf.values()):12.0f} {100*extra/sum(tf.values()):12.2f}%")
    print("READING: at this width the converter+storage terms are orders of")
    print("magnitude below the matrix terms, so they do NOT decide the verdict.")
    print("They would matter at large H with a cheap readout -- record the scale,")
    print("do not claim they are free in general.")

    print()
    print("=" * 78)
    print("PART 5 -- per-term dump, full_event datapath, default unpriced costs")
    print("=" * 78)
    for lab in sorted(groups):
        ds = groups[lab]
        d0 = ds[0]
        rz, _ = mean_sd([d["rate_emitted"] for d in ds])
        rs, _ = mean_sd([d["rate_state"] for d in ds])
        t = terms(d0["variant"], V, E, H, rz, rs, "full_event")
        tot = sum(t.values())
        parts = "  ".join(f"{k}={v:.0f}" for k, v in t.items())
        print(f"{lab:22s} tot={tot:8.0f}  {parts}")


if __name__ == "__main__":
    main()
