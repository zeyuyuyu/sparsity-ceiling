"""Exp2: dissociate MEMORY vs DIFFICULTY.
Task A = copy (memory load N, easy content). Task B = single-step classification under a
frozen random-MLP teacher (NO memory, difficulty = #classes C). Predict rho_min rises with N
but stays low/flat in C."""
import json,argparse,torch,torch.nn as nn
import snntorch as snn; from snntorch import surrogate
def A():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--H",type=int,default=256);p.add_argument("--E",type=int,default=32)
    p.add_argument("--S",type=int,default=6);p.add_argument("--D",type=int,default=64)
    p.add_argument("--steps",type=int,default=2500);p.add_argument("--bs",type=int,default=128)
    p.add_argument("--lr",type=float,default=2e-3)
    p.add_argument("--out",default="/work/zeyuwang/neuro_poc/paper_runs/exp2_dissoc.json");return p.parse_args()
# ---- Task A: copy ----
def cbatch(N,S,bs,dev):
    x=torch.randint(1,S+1,(bs,N),device=dev);d=torch.full((bs,1),S+1,device=dev)
    b=torch.zeros((bs,N),dtype=torch.long,device=dev);inp=torch.cat([x,d,b],1)
    tgt=torch.cat([torch.zeros((bs,N+1),dtype=torch.long,device=dev),x],1)
    m=torch.cat([torch.zeros(bs,N+1,device=dev),torch.ones(bs,N,device=dev)],1);return inp,tgt,m
class CopyNet(nn.Module):
    def __init__(s,V,E,H):
        super().__init__();s.H=H;s.emb=nn.Embedding(V,E);s.Wi=nn.Linear(E,H);s.Wr=nn.Linear(H,H,bias=False)
        s.ro=nn.Linear(H,V);s.lif=snn.Leaky(beta=0.9,spike_grad=surrogate.fast_sigmoid(slope=25))
    def forward(s,x):
        B,L=x.shape;h=torch.zeros(B,s.H,device=x.device);mem=s.lif.init_leaky();lo=[];f=0.;r=0.
        for t in range(L):
            c=s.Wi(s.emb(x[:,t]))+s.Wr(h);h,mem=s.lif(c,mem);f+=h.mean().item();r=r+h.mean();lo.append(s.ro(h))
        return torch.stack(lo,1),f/L,r/L
def run_copy(N,tgt,a,dev,lam=2.0):
    V=a.S+2;net=CopyNet(V,a.E,a.H).to(dev);opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    ce=nn.CrossEntropyLoss(reduction="none")
    for it in range(a.steps):
        inp,t,m=cbatch(N,a.S,a.bs,dev);log,_,r=net(inp)
        l=(ce(log.reshape(-1,V),t.reshape(-1))*m.reshape(-1)).sum()/m.sum()
        if it>a.steps//5:l=l+lam*(r-tgt).abs()
        opt.zero_grad();l.backward();opt.step()
    net.eval();ac=0.;fr=0.
    with torch.no_grad():
        for _ in range(15):
            inp,t,m=cbatch(N,a.S,a.bs,dev);log,f,_=net(inp);ac+=(((log.argmax(-1)==t).float()*m).sum()/m.sum()).item();fr+=f
    return ac/15,fr/15
# ---- Task B: single-step classification, frozen random teacher (no memory) ----
class Teacher(nn.Module):
    def __init__(s,D,C):
        super().__init__();s.net=nn.Sequential(nn.Linear(D,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,C))
        for p in s.parameters():p.requires_grad=False
    def forward(s,x):return s.net(x).argmax(-1)
class FFNet(nn.Module):  # recurrent net run for 1 step = feedforward, vector input
    def __init__(s,D,H,C):
        super().__init__();s.H=H;s.Wi=nn.Linear(D,H);s.ro=nn.Linear(H,C)
        s.lif=snn.Leaky(beta=0.9,spike_grad=surrogate.fast_sigmoid(slope=25))
    def forward(s,x):
        mem=s.lif.init_leaky();h,mem=s.lif(s.Wi(x),mem);return s.ro(h),h.mean().item(),h.mean()
def run_ffwd(C,tgt,a,dev,lam=2.0):
    torch.manual_seed(C);teacher=Teacher(a.D,C).to(dev)
    net=FFNet(a.D,a.H,C).to(dev);opt=torch.optim.Adam(net.parameters(),lr=a.lr);ce=nn.CrossEntropyLoss()
    for it in range(a.steps):
        x=torch.randn(a.bs,a.D,device=dev);y=teacher(x)
        log,_,r=net(x);l=ce(log,y)
        if it>a.steps//5:l=l+lam*(r-tgt).abs()
        opt.zero_grad();l.backward();opt.step()
    net.eval();ac=0.;fr=0.
    with torch.no_grad():
        for _ in range(15):
            x=torch.randn(a.bs,a.D,device=dev);y=teacher(x);log,f,_=net(x);ac+=(log.argmax(-1)==y).float().mean().item();fr+=f
    return ac/15,fr/15
def main():
    a=A();dev=torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    grid=[0.03,0.05,0.08,0.15];out={"memory_task":[],"difficulty_task":[]}
    for N in [4,6,8]:
        cur=[]
        for g in grid:
            ac,fr=run_copy(N,g,a,dev);cur.append({"target":g,"firing":round(fr,4),"acc":round(ac,3)})
            print("[MEM] N=%d tgt=%.2f fire=%.3f acc=%.3f"%(N,g,fr,ac),flush=True)
        out["memory_task"].append({"N":N,"curve":cur})
    for C in [10,40,160]:
        cur=[]
        for g in grid:
            ac,fr=run_ffwd(C,g,a,dev);cur.append({"target":g,"firing":round(fr,4),"acc":round(ac,3)})
            print("[DIFF] C=%d tgt=%.2f fire=%.3f acc=%.3f"%(C,g,fr,ac),flush=True)
        out["difficulty_task"].append({"C":C,"curve":cur})
    json.dump(out,open(a.out,"w"),indent=2);print("SAVED",a.out,flush=True)
if __name__=="__main__":main()
