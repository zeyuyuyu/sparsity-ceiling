"""
Neuromorphic startup PoC — v0, sparse spiking perception vs conventional CNN.

Thesis under test: a spiking neural network (SNN) reaches accuracy comparable to a
matched conventional CNN (ANN) while doing FAR fewer synaptic operations, because
spikes are sparse. Sparse ops -> low energy -> the reason neuromorphic wins at the
power/latency-constrained edge (robot/drone perception). We MEASURE it, not assert.

Data: FashionMNIST (local, no download), rate-coded into T-step spike trains
(the standard snntorch PoC setup). Honest caveat: rate-coded static images are a
stand-in for native event-camera streams (N-MNIST/DVS-Gesture) — same algorithmic
value prop, lighter data path. N-MNIST is the documented next step.

Outputs JSON + printed table: test accuracy (SNN vs ANN), mean firing rate
(sparsity), and an energy proxy (ANN MACs @4.6pJ vs SNN SynOps @0.9pJ, 45nm Horowitz).
"""
import json, time, argparse
import torch, torch.nn as nn
import torchvision, torchvision.transforms as TT
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

DATA_ROOT = "/work/zeyuwang/hpc_rebuttal/data"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/work/zeyuwang/neuro_poc/poc_results.json")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam", type=float, default=0.0, help="sparsity reg weight")
    p.add_argument("--target", type=float, default=-1.0, help="target firing rate; <0 = old one-sided L1")
    p.add_argument("--warmup", type=int, default=2, help="epochs before reg kicks in")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()

def loaders(bs):
    tf = TT.ToTensor()
    tr = torchvision.datasets.FashionMNIST(DATA_ROOT, train=True,  download=False, transform=tf)
    te = torchvision.datasets.FashionMNIST(DATA_ROOT, train=False, download=False, transform=tf)
    return (DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=4, drop_last=True),
            DataLoader(te, batch_size=bs, shuffle=False, num_workers=4))

# dense MAC counts (one forward, one timestep), input 28x28, conv 5x5 valid, pool 2
h1=28-5+1; h1p=h1//2          # 24 -> 12
h2=h1p-5+1; h2p=h2//2         # 8  -> 4
MACS={"conv1": h1*h1*12*(1*5*5),
      "conv2": h2*h2*32*(12*5*5),
      "fc":    (32*h2p*h2p)*10}
TOTAL_MAC=sum(MACS.values())

class SNNNet(nn.Module):
    def __init__(self,T):
        super().__init__(); self.T=T
        sg=surrogate.fast_sigmoid(slope=25)
        self.c1=nn.Conv2d(1,12,5);  self.l1=snn.Leaky(beta=0.9,spike_grad=sg)
        self.p1=nn.MaxPool2d(2)
        self.c2=nn.Conv2d(12,32,5); self.l2=snn.Leaky(beta=0.9,spike_grad=sg)
        self.p2=nn.MaxPool2d(2)
        self.fc=nn.Linear(32*h2p*h2p,10); self.l3=snn.Leaky(beta=0.9,spike_grad=sg)
    def forward(self,x):  # x static [B,1,28,28] in [0,1]
        m1=self.l1.init_leaky(); m2=self.l2.init_leaky(); m3=self.l3.init_leaky()
        spk_out=[]; fr={"in":0.,"s1":0.,"s2":0.}; r1=0.; r2=0.
        for t in range(self.T):
            xt=torch.bernoulli(x.clamp(0,1))     # rate coding -> input spikes
            fr["in"]+=xt.mean().item()
            s1,m1=self.l1(self.p1(self.c1(xt)),m1); fr["s1"]+=s1.mean().item(); r1=r1+s1.mean()
            s2,m2=self.l2(self.p2(self.c2(s1)),m2); fr["s2"]+=s2.mean().item(); r2=r2+s2.mean()
            s3,m3=self.l3(self.fc(s2.flatten(1)),m3); spk_out.append(s3)
        for k in fr: fr[k]/=self.T
        return torch.stack(spk_out), fr, (r1/self.T, r2/self.T)  # mean per-layer rates (tensors)

class ANNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,12,5),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(12,32,5),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(32*h2p*h2p,10))
    def forward(self,x): return self.net(x)

def run_snn(a,dev,tr,te):
    net=SNNNet(a.T).to(dev); opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    lf=SF.ce_rate_loss()
    for ep in range(a.epochs):
        net.train(); t0=time.time(); last=0.
        on = ep >= a.warmup            # reg warmup: let net learn to fire first
        for x,y in tr:
            x,y=x.to(dev),y.to(dev)
            spk,_,(r1,r2)=net(x); loss=lf(spk,y)
            if a.lam>0 and on:
                if a.target>=0:        # two-sided: penalize deviation from target rate (anti-collapse)
                    reg=(r1-a.target).abs()+(r2-a.target).abs()
                else:                  # one-sided L1 down (old, collapses)
                    reg=r1+r2
                loss=loss+a.lam*reg
            opt.zero_grad(); loss.backward(); opt.step(); last=loss.item()
        print(f"[SNN lam={a.lam} tgt={a.target}] ep{ep+1}/{a.epochs} loss{last:.3f} {time.time()-t0:.1f}s",flush=True)
    net.eval(); c=0;n=0; frs={"in":0.,"s1":0.,"s2":0.};nb=0
    with torch.no_grad():
        for x,y in te:
            x,y=x.to(dev),y.to(dev); spk,fr,_=net(x)
            c+=(spk.sum(0).argmax(1)==y).sum().item(); n+=y.numel()
            for k in frs: frs[k]+=fr[k]
            nb+=1
    for k in frs: frs[k]/=nb
    synops=(MACS["conv1"]*frs["in"]+MACS["conv2"]*frs["s1"]+MACS["fc"]*frs["s2"])*a.T
    return c/n, frs, synops

def run_ann(a,dev,tr,te):
    net=ANNNet().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    lf=nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        net.train(); t0=time.time(); last=0.
        for x,y in tr:
            x,y=x.to(dev),y.to(dev); out=net(x); loss=lf(out,y)
            opt.zero_grad(); loss.backward(); opt.step(); last=loss.item()
        print(f"[ANN] ep{ep+1}/{a.epochs} loss{last:.3f} {time.time()-t0:.1f}s",flush=True)
    net.eval(); c=0;n=0
    with torch.no_grad():
        for x,y in te:
            x,y=x.to(dev),y.to(dev); c+=(net(x).argmax(1)==y).sum().item(); n+=y.numel()
    return c/n

def main():
    a=get_args()
    dev=torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    print("device",dev,"| dense MACs/inf:",f"{TOTAL_MAC:,}",flush=True)
    tr,te=loaders(a.bs)
    E_MAC,E_AC=4.6e-12,0.9e-12
    ann=run_ann(a,dev,tr,te)
    snn_acc,frs,synops=run_snn(a,dev,tr,te)
    e_ann=TOTAL_MAC*E_MAC; e_snn=synops*E_AC
    res={"task":"FashionMNIST (rate-coded spikes) — event-vision PoC",
         "T":a.T,"epochs":a.epochs,"sparsity_lambda":a.lam,"sparsity_target":a.target,
         "ann_test_acc":round(ann,4),"snn_test_acc":round(snn_acc,4),
         "snn_firing_rates":{k:round(v,4) for k,v in frs.items()},
         "dense_MACs_per_inf":TOTAL_MAC,"snn_SynOps_per_inf":int(synops),
         "op_reduction_x":round(TOTAL_MAC/max(synops,1),2),
         "energy_ANN_uJ":round(e_ann*1e6,4),"energy_SNN_uJ":round(e_snn*1e6,4),
         "energy_reduction_x":round(e_ann/max(e_snn,1e-18),2)}
    with open(a.out,"w") as f: json.dump(res,f,indent=2)
    print("\n===== NEUROMORPHIC PoC RESULT =====")
    for k,v in res.items(): print(f"{k:22s}: {v}")
    print("saved ->",a.out,flush=True)

if __name__=="__main__":
    main()
