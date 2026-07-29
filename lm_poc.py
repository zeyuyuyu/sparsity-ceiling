"""
Neuromorphic-runs-LLM exploration (small-scale, honest).

NOT a 7B LLM (that needs frontier scale + hits the memory wall). This is the
tractable analog the literature actually demos (SpiNNaker EGRU, Loihi 370M):
a SPIKING recurrent language model vs an architecturally IDENTICAL continuous RNN.
Same embedding / input transform / recurrent weight / readout — the ONLY difference
is the hidden neuron: continuous tanh (ANN) vs leaky-integrate-fire spikes (SNN).
So any energy gap is attributable to spiking alone, not to a lighter model.

Char-level language modeling on WikiText-103 (local arrow, no download).
Measures: bits-per-char (quality), hidden firing rate (sparsity), energy proxy
(dense MACs @4.6pJ vs spike-driven ACs @0.9pJ, 45nm Horowitz).
"""
import json, time, math, argparse
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate

ARROW="/work/zeyuwang/5duaa/data_cache/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow"

def get_args():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--E",type=int,default=64)      # embed dim
    p.add_argument("--H",type=int,default=256)     # hidden
    p.add_argument("--L",type=int,default=128)     # seq / timesteps
    p.add_argument("--bs",type=int,default=64)
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--lr",type=float,default=2e-3)
    p.add_argument("--lam",type=float,default=0.0) # spike sparsity target reg
    p.add_argument("--target",type=float,default=0.10)
    p.add_argument("--chars",type=int,default=1_400_000)
    p.add_argument("--out",default="/work/zeyuwang/neuro_poc/lm_results.json")
    return p.parse_args()

def load_text(nchars):
    from datasets import Dataset
    d=Dataset.from_file(ARROW)
    txt="\n".join(t for t in d["text"] if t and t.strip())
    return txt[:nchars]

def make_data(txt,L,dev):
    chars=sorted(set(txt)); V=len(chars)
    stoi={c:i for i,c in enumerate(chars)}
    ids=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
    n=(len(ids)-1)//L
    x=ids[:n*L].view(n,L)
    y=ids[1:n*L+1].view(n,L)
    cut=int(n*0.9)
    return (x[:cut],y[:cut]),(x[cut:],y[cut:]),V

class RecLM(nn.Module):
    """Shared architecture; mode='ann' (tanh) or 'snn' (LIF)."""
    def __init__(self,V,E,H,mode):
        super().__init__(); self.H=H; self.mode=mode
        self.emb=nn.Embedding(V,E)
        self.W_in=nn.Linear(E,H,bias=True)     # dense (continuous embedding) -> MAC
        self.W_rec=nn.Linear(H,H,bias=False)   # recurrent -> spike-driven AC in SNN
        self.readout=nn.Linear(H,V)            # spike-driven AC in SNN
        if mode=="snn":
            self.lif=snn.Leaky(beta=0.9,spike_grad=surrogate.fast_sigmoid(slope=25))
    def forward(self,x):                       # x [B,L]
        B,L=x.shape
        h=torch.zeros(B,self.H,device=x.device)
        if self.mode=="snn": mem=self.lif.init_leaky()
        logits=[]; fire=0.0; r_acc=0.0
        for t in range(L):
            e=self.emb(x[:,t])
            cur=self.W_in(e)+self.W_rec(h)
            if self.mode=="snn":
                h,mem=self.lif(cur,mem); fire+=h.mean().item(); r_acc=r_acc+h.mean()
            else:
                h=torch.tanh(cur)
            logits.append(self.readout(h))
        return torch.stack(logits,1), fire/L, (r_acc/L if self.mode=="snn" else None)

def run(mode,a,dev,tr,va,V):
    net=RecLM(V,a.E,a.H,mode).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    ce=nn.CrossEntropyLoss()
    xt,yt=tr; xv,yv=va
    nb=(xt.size(0)+a.bs-1)//a.bs
    for ep in range(a.epochs):
        net.train(); t0=time.time(); perm=torch.randperm(xt.size(0)); last=0.
        for i in range(nb):
            idx=perm[i*a.bs:(i+1)*a.bs]
            x=xt[idx].to(dev); y=yt[idx].to(dev)
            log,_,r=net(x)
            loss=ce(log.reshape(-1,V),y.reshape(-1))
            if mode=="snn" and a.lam>0 and ep>=1:
                loss=loss+a.lam*(r-a.target).abs()
            opt.zero_grad(); loss.backward(); opt.step(); last=loss.item()
        print(f"[{mode}] ep{ep+1}/{a.epochs} loss{last:.3f} {time.time()-t0:.1f}s",flush=True)
    # eval
    net.eval(); tot=0.; ntok=0; fr=0.; nb_v=0
    with torch.no_grad():
        for i in range((xv.size(0)+a.bs-1)//a.bs):
            x=xv[i*a.bs:(i+1)*a.bs].to(dev); y=yv[i*a.bs:(i+1)*a.bs].to(dev)
            log,f,_=net(x)
            tot+=ce(log.reshape(-1,V),y.reshape(-1)).item()*y.numel(); ntok+=y.numel()
            fr+=f; nb_v+=1
    ce_nats=tot/ntok
    return {"bpc":ce_nats/math.log(2),"ppl":math.exp(ce_nats),"firing":fr/nb_v}

def energy(a,V,r):
    E,H=a.E,a.H
    Emac,Eac=4.6e-12,0.9e-12
    # per-timestep op counts
    win=E*H                 # dense input transform (both models)
    wrec=H*H
    rdo=H*V
    ann_ops=(win+wrec+rdo)  # all dense MAC
    e_ann=ann_ops*Emac*a.L
    # SNN: W_in dense MAC (continuous embedding); W_rec + readout spike-driven AC @ firing r
    e_snn=(win*Emac + (wrec+rdo)*r*Eac)*a.L
    return e_ann,e_snn,ann_ops,win,wrec,rdo

def main():
    a=get_args()
    dev=torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    txt=load_text(a.chars); tr,va,V=make_data(txt,a.L,dev)
    print(f"device {dev} | chars {len(txt):,} | vocab {V} | train seqs {tr[0].size(0)}",flush=True)
    ann=run("ann",a,dev,tr,va,V)
    snn_=run("snn",a,dev,tr,va,V)
    e_ann,e_snn,ann_ops,win,wrec,rdo=energy(a,V,snn_["firing"])
    res={"task":"WikiText-103 char-level LM — spiking-RNN vs matched tanh-RNN",
         "vocab":V,"embed":a.E,"hidden":a.H,"seq_len":a.L,"epochs":a.epochs,
         "sparsity_lambda":a.lam,"sparsity_target":a.target if a.lam>0 else None,
         "ann_bpc":round(ann["bpc"],4),"snn_bpc":round(snn_["bpc"],4),
         "ann_ppl":round(ann["ppl"],2),"snn_ppl":round(snn_["ppl"],2),
         "snn_firing_rate":round(snn_["firing"],4),
         "ann_ops_per_step":int(ann_ops),
         "input_floor_frac_of_snn_energy":round(win*4.6e-12*a.L/e_snn,3),
         "energy_ANN_nJ":round(e_ann*1e9,3),"energy_SNN_nJ":round(e_snn*1e9,3),
         "energy_reduction_x":round(e_ann/max(e_snn,1e-18),2)}
    with open(a.out,"w") as f: json.dump(res,f,indent=2)
    print("\n===== NEUROMORPHIC LM EXPLORATION =====")
    for k,v in res.items(): print(f"{k:30s}: {v}")
    print("saved ->",a.out,flush=True)

if __name__=="__main__": main()
