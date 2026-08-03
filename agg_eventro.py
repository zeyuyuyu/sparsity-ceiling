"""Aggregate the event-driven-readout row and report pJ under both datapaths.

Published proxy (`energy_pJ_per_token` in the JSON) now prices W_out at
H*V*r_out*E_AC whenever --out_theta > 0.  The `full_event` column additionally
prices W_in at the event rate and charges the analog converter/storage terms, so
it is the honest best case for a fully event-driven datapath.
"""
import json
import sys

E_MAC, E_AC = 4.6, 0.9
V, E, H = 284, 64, 256
WIN, WMIX, WOUT = E * H, H * H, H * V
R = "ssm3way_runs/"
OTS = ("0.02", "0.05", "0.1", "0.25", "0.5", "1.0", "2.0")
STEM = {"analog": "analog_charlm_s0_theta0.15",
        "digital": "digital_charlm_s0_reg_n0.02"}


def full_event(v, rz, rs, ro):
    """pJ/token, input+readout event-priced, analog converters+storage charged."""
    if v == "digital":
        t = {"W_in": WIN * E_AC, "state": H * E_MAC, "W_mix": WMIX * E_MAC}
    else:
        t = {"W_in": WIN * rz * E_AC, "state": H * E_AC, "W_mix": WMIX * rz * E_AC}
        t["cv"] = H * rz * 0.06 + H * 0.02
    t["W_out"] = WOUT * ro * E_AC
    return sum(t.values())


def row(arm, ot, d, dref):
    pj = d["energy_pJ_per_token"]
    ro = d.get("rate_out", 1.0)
    fe = full_event(arm, d["rate_emitted"], d["rate_state"], ro)
    print("%-8s %5s %7.4f %+7.4f %6.3f %6.3f %9.0f %5.2fx %10.0f %5.2fx"
          % (arm, ot, d["bpc"], d["bpc"] - dref, ro, d["rate_emitted"],
             pj, 712448 / pj, fe, 712448 / fe))


def main():
    refs = {k: json.load(open(R + v + ".json")) for k, v in STEM.items()}
    dref = refs["digital"]["bpc"]
    print("reference = digital + state-noise 0.02 (the post-retraction correct "
          "comparator), bpc %.4f, 712448 pJ/token published" % dref)
    print("%-8s %5s %7s %7s %6s %6s %9s %6s %10s %6s"
          % ("arm", "ot", "bpc", "dbpc", "r_out", "r_z", "pJ_pub", "x", "pJ_fev", "x"))
    for arm in ("analog", "digital"):
        row(arm, "0", refs[arm], dref)
        for ot in OTS:
            try:
                d = json.load(open("%s%s_ot%s.json" % (R, STEM[arm], ot)))
            except FileNotFoundError:
                print("%-8s %5s MISSING" % (arm, ot))
                continue
            row(arm, ot, d, dref)
        print()


if __name__ == "__main__":
    main()
