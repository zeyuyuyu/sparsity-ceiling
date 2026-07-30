"""
Are neuromorphic chips suitable for SSMs -- and is the NON-SPIKING (analog-state)
route the right fit?  (N. Imam TODO; direction #1 of "The Sparsity Ceiling".)

Controlled 4-way comparison.  ONE SSM, PARAMETER-MATCHED: every variant owns
exactly the same tensors -- emb(V,E), W_in(E,H), diagonal decay a(H), W_mix(H,H),
W_out(H,V).  The only thing that moves is WHERE the nonlinearity/binarization sits:

  core (variants 1-3):  h_t = a*h_{t-1} + W_in e_t          <- linear, CONTINUOUS state
  head:                 logits = W_out( W_mix z_t )

  (1) digital     z_t = GELU(h_t)                 continuous state, continuous output
                                                  == S4D-real baseline on GPU
  (2) spikeout    z_t = LIF(h_t)                  continuous state, SPIKING output
                                                  == SPikE-SSM-style
  (3) analog      z_t = h_t * delta_event_mask    ANALOG state (leak + noise + 6-bit
                                                  rails), send-on-delta graded events
  (4) spikestate  h_t = LIF(a*h_{t-1} + W_in e_t + W_mix h_{t-1});  logits = W_out h_t
                  the FLOOR CONTROL: recurrent state itself carried in spikes.
                  This is the regime the firing-floor bound rho >= H_b^-1(log2 M / H)
                  applies to.  W_mix is reused as the recurrent matrix so the
                  parameter count stays identical -- structurally this is the
                  canonical form of its class, not a crippled digital SSM.

HYPOTHESIS: (3) matches (1)'s quality while its activity sits FAR below the ~50%
firing floor that (4) hits; (2)'s sparsity lives only in the output, not the state.
If (3) does NOT beat the floor, that is the result and it gets reported as such.

Measures: quality (bpc / copy accuracy), emitted activity rate, state activity rate,
energy proxy (45nm Horowitz: dense MAC 4.6 pJ vs accumulate 0.9 pJ).  For (3) the
graded analog event is priced BOTH ways -- optimistic (AC) and conservative (MAC) --
because a graded event is not a 1-bit spike.

Tasks: char-level WikiText-103 (local arrow) | synthetic copy (explicit memory load).
"""
import json, math, time, argparse
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate

ARROW = ("/work/zeyuwang/5duaa/data_cache/wikitext/wikitext-103-raw-v1/0.0.0/"
         "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow")
E_MAC, E_AC = 4.6e-12, 0.9e-12
VARIANTS = ("digital", "spikeout", "analog", "spikestate")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--task", choices=("charlm", "copy"), default="charlm")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--E", type=int, default=64)
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam", type=float, default=0.0, help="two-sided target-rate reg")
    p.add_argument("--target", type=float, default=0.10)
    p.add_argument("--theta", type=float, default=0.10, help="analog send-on-delta threshold")
    p.add_argument("--noise", type=float, default=0.02, help="analog state noise sigma")
    p.add_argument("--bits", type=int, default=6, help="analog state precision")
    p.add_argument("--rail", type=float, default=4.0, help="analog state clip")
    p.add_argument("--chars", type=int, default=1_400_000)
    p.add_argument("--copy_k", type=int, default=16, help="copy-task token alphabet")
    p.add_argument("--copy_n", type=int, default=20000, help="copy-task sequences")
    p.add_argument("--out", required=True)
    return p.parse_args()


# ---------------------------------------------------------------- data
def data_charlm(a):
    from datasets import Dataset
    d = Dataset.from_file(ARROW)
    txt = "\n".join(t for t in d["text"] if t and t.strip())[:a.chars]
    chars = sorted(set(txt)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    n = (len(ids) - 1) // a.L
    x = ids[:n * a.L].view(n, a.L); y = ids[1:n * a.L + 1].view(n, a.L)
    cut = int(n * 0.9)
    mask = torch.ones_like(y, dtype=torch.bool)
    return (x[:cut], y[:cut], mask[:cut]), (x[cut:], y[cut:], mask[cut:]), V


def data_copy(a):
    """[s_1..s_K, DELIM, s_1..s_K] -> next-token; loss only on the recalled half."""
    K = a.copy_k; V = K + 1; DELIM = K
    half = (a.L - 1) // 2
    s = torch.randint(0, K, (a.copy_n, half))
    seq = torch.cat([s, torch.full((a.copy_n, 1), DELIM), s], 1)   # [n, 2*half+1]
    x, y = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros_like(y, dtype=torch.bool); mask[:, half:] = True
    cut = int(a.copy_n * 0.9)
    return (x[:cut], y[:cut], mask[:cut]), (x[cut:], y[cut:], mask[cut:]), V


# ---------------------------------------------------------------- model
class SSM(nn.Module):
    def __init__(self, V, E, H, variant, a):
        super().__init__()
        self.H, self.variant, self.cfg = H, variant, a
        self.emb = nn.Embedding(V, E)
        self.W_in = nn.Linear(E, H, bias=True)
        self.log_dt = nn.Parameter(torch.linspace(math.log(1e-3), math.log(1e-1), H))
        self.W_mix = nn.Linear(H, H, bias=False)
        self.W_out = nn.Linear(H, V)
        if variant in ("spikeout", "spikestate"):
            self.lif = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(slope=25))

    def decay(self):
        return torch.exp(-torch.exp(self.log_dt))          # diagonal a in (0,1)

    def _analog(self, h):
        """sub-threshold analog state: rails, additive noise, finite precision."""
        c = self.cfg
        h = h.clamp(-c.rail, c.rail)
        if self.training and c.noise > 0:
            h = h + torch.randn_like(h) * c.noise
        if c.bits > 0:
            q = 2 * c.rail / (2 ** c.bits)
            h = h + (torch.round(h / q) * q - h).detach()   # straight-through
        return h

    def forward(self, x):
        B, L = x.shape
        dev = x.device
        h = torch.zeros(B, self.H, device=dev)
        href = torch.zeros(B, self.H, device=dev)           # analog last-sent value
        mem = self.lif.init_leaky() if self.variant in ("spikeout", "spikestate") else None
        a = self.decay()
        logits, z_act, s_act = [], [], []
        for t in range(L):
            e = self.emb(x[:, t])
            if self.variant == "spikestate":
                cur = a * h + self.W_in(e) + self.W_mix(h)
                h, mem = self.lif(cur, mem)
                s_act.append(h.mean()); z_act.append(h.mean())
                logits.append(self.W_out(h))
                continue
            h = a * h + self.W_in(e)
            if self.variant == "analog":
                h = self._analog(h)
            s_act.append((h.abs() > 1e-6).float().mean())
            if self.variant == "digital":
                z = torch.nn.functional.gelu(h)
                z_act.append(torch.ones((), device=dev))
            elif self.variant == "spikeout":
                z, mem = self.lif(h, mem)
                z_act.append(z.mean())
            else:                                            # analog send-on-delta
                m = ((h - href).abs() > self.cfg.theta).float()
                href = href * (1 - m) + h.detach() * m
                z = h * m.detach()
                z_act.append(m.mean())
            logits.append(self.W_out(self.W_mix(z)))
        return (torch.stack(logits, 1),
                torch.stack(z_act).mean(), torch.stack(s_act).mean())


# ---------------------------------------------------------------- energy
def energy(a, V, r_z, r_s):
    """pJ per token, per-term, explicit about every assumption."""
    E, H = a.E, a.H
    win = E * H; wmix = H * H; wout = H * V
    terms = {}
    if a.variant == "digital":
        terms = {"W_in_MAC": win * E_MAC, "state_MAC": H * E_MAC,
                 "W_mix_MAC": wmix * E_MAC, "W_out_MAC": wout * E_MAC}
        tot = sum(terms.values()); tot_cons = tot
    elif a.variant == "spikeout":
        terms = {"W_in_MAC": win * E_MAC, "state_MAC": H * E_MAC,
                 "W_mix_AC_at_rz": wmix * r_z * E_AC, "W_out_MAC": wout * E_MAC}
        tot = sum(terms.values()); tot_cons = tot
    elif a.variant == "analog":
        # analog leak/integrate is physical -> priced at AC.  graded event priced
        # optimistically (AC) and conservatively (MAC); truth is in between.
        terms = {"W_in_MAC": win * E_MAC, "state_analog_AC": H * E_AC,
                 "W_mix_event_at_rz": wmix * r_z * E_AC, "W_out_MAC": wout * E_MAC}
        tot = sum(terms.values())
        tot_cons = win * E_MAC + H * E_AC + wmix * r_z * E_MAC + wout * E_MAC
    else:                                                    # spikestate
        terms = {"W_in_MAC": win * E_MAC, "state_MAC": H * E_MAC,
                 "W_mix_AC_at_rs": wmix * r_s * E_AC, "W_out_AC_at_rs": wout * r_s * E_AC}
        tot = sum(terms.values()); tot_cons = tot
    return {k: round(v * 1e12, 4) for k, v in terms.items()}, tot * 1e12, tot_cons * 1e12


# ---------------------------------------------------------------- train / eval
def evaluate(net, va, V, a, dev):
    x, y, m = va
    net.eval(); tot = 0.0; ntok = 0; corr = 0; rz = 0.0; rs = 0.0; nb = 0
    ce = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for i in range(0, x.size(0), a.bs):
            xb = x[i:i + a.bs].to(dev); yb = y[i:i + a.bs].to(dev); mb = m[i:i + a.bs].to(dev)
            log, z, s = net(xb)
            l = ce(log.reshape(-1, V), yb.reshape(-1)).view_as(yb)
            tot += (l * mb).sum().item(); ntok += mb.sum().item()
            corr += ((log.argmax(-1) == yb) & mb).sum().item()
            rz += z.item(); rs += s.item(); nb += 1
    nats = tot / ntok
    return {"bpc": nats / math.log(2), "ppl": math.exp(min(nats, 20)),
            "acc": corr / ntok, "rate_emitted": rz / nb, "rate_state": rs / nb}


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    tr, va, V = data_charlm(a) if a.task == "charlm" else data_copy(a)
    net = SSM(V, a.E, a.H, a.variant, a).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"[{a.variant}/{a.task}] dev {dev} vocab {V} params {nparam:,} "
          f"train {tr[0].size(0)} seqs", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss(reduction="none")
    xt, yt, mt = tr
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); perm = torch.randperm(xt.size(0)); last = 0.0
        for i in range(0, xt.size(0), a.bs):
            idx = perm[i:i + a.bs]
            xb = xt[idx].to(dev); yb = yt[idx].to(dev); mb = mt[idx].to(dev)
            log, z, s = net(xb)
            l = ce(log.reshape(-1, V), yb.reshape(-1)).view_as(yb)
            loss = (l * mb).sum() / mb.sum()
            if a.lam > 0 and ep >= 1 and a.variant != "digital":
                loss = loss + a.lam * (z - a.target).abs()
            opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        print(f"[{a.variant}] ep{ep+1}/{a.epochs} loss {last:.4f} "
              f"{time.time()-t0:.1f}s", flush=True)
    ev = evaluate(net, va, V, a, dev)
    terms, tot, tot_cons = energy(a, V, ev["rate_emitted"], ev["rate_state"])
    res = {"variant": a.variant, "task": a.task, "seed": a.seed, "params": nparam,
           "vocab": V, "E": a.E, "H": a.H, "L": a.L, "epochs": a.epochs, "lr": a.lr,
           "lam": a.lam, "target": a.target if a.lam > 0 else None,
           "analog": {"theta": a.theta, "noise": a.noise, "bits": a.bits, "rail": a.rail}
           if a.variant == "analog" else None,
           "bpc": round(ev["bpc"], 4), "ppl": round(ev["ppl"], 3),
           "acc": round(ev["acc"], 4),
           "rate_emitted": round(ev["rate_emitted"], 4),
           "rate_state": round(ev["rate_state"], 4),
           "energy_pJ_per_token": round(tot, 3),
           "energy_pJ_per_token_conservative": round(tot_cons, 3),
           "energy_terms_pJ": terms}
    with open(a.out, "w") as f: json.dump(res, f, indent=2)
    print("\n===== SSM 4-WAY CELL =====")
    for k, v in res.items(): print(f"{k:34s}: {v}")
    print("saved ->", a.out, flush=True)


if __name__ == "__main__":
    main()
