"""Clean rho_min(N): trace accuracy-vs-firing curves per memory load N.
For each N and each target firing rho*, train the spiking copy net to that target
and record (achieved_firing, accuracy). rho_min(N) = min achieved firing with acc>=tau.
Prediction (theory): the acc-vs-firing curve shifts right as N grows -> floor rises."""
import json,argparse,torch,torch.nn as nn
import snntorch as snn; from snntorch import surrogate
def A():
    p=argparse.ArgumentParser(); p.add_argument("--gpu",type=int,default=1)
    p.add_argument("--H",type=int,default=256);p.add_argument("--E",type=int,default=32)
    p.add_argument("--S",type=int,default=6);p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--bs",type=int,default=128);p.add_argument("--lr",type=float,default=2e-3)
    p.add_argument("--out",default="/work/zeyuwang/neuro_poc/paper_runs/mem2.json");return p.parse_args()
def batch(N,S,bs,dev):
    x=torch.randint(1,S+1,(bs,N),device=dev);d=torch.full((bs,1),S+1,device=dev)
    b=torch.zeros((bs,N),dtype=torch.long,device=dev);inp=torch.cat([x,d,b],1)
    tgt=torch.cat([torch.zeros((bs,N+1),dtype=torch.long,device=dev),x],1)
    m=torch.cat([torch.zeros(bs,N+1,device=dev),torch.ones(bs,N,device=dev)],1);return inp,tgt,m
class Net(nn.Module):
    def __init__(s,V,E,H,mode):
        super().__init__();s.H=H;s.mode=mode;s.emb=nn.Embedding(V,E)
        s.Wi=nn.Linear(E,H);s.Wr=nn.Linear(H,H,bias=False);s.ro=nn.Linear(H,V)
        if mode=="snn":s.lif=snn.Leaky(beta=0.9,spike_grad=surrogate.fast_sigmoid(slope=25))
    def forward(s,x):
        B,L=x.shape;h=torch.zeros(B,s.H,device=x.device)
        if s.mode=="snn":mem=s.lif.init_leaky()
        lo=[];f=0.;r=0.
        for t in range(L):
            c=s.Wi(s.emb(x[:,t]))+s.Wr(h)
            if s.mode=="snn":h,mem=s.lif(c,mem);f+=h.mean().item();r=r+h.mean()
            else:h=torch.tanh(c)
            lo.append(s.ro(h))
        return torch.stack(lo,1),f/L,(r/L if s.mode=="snn" else None)
def tr(N,mode,a,dev,tgt_rate=None,lam=2.0):
    V=a.S+2;net=Net(V,a.E,a.H,mode).to(dev);opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    ce=nn.CrossEntropyLoss(reduction="none")
    for it in range(a.steps):
        inp,t,m=batch(N,a.S,a.bs,dev);log,_,r=net(inp)
        l=(ce(log.reshape(-1,V),t.reshape(-1))*m.reshape(-1)).sum()/m.sum()
        if mode=="snn" and tgt_rate is not None and it>a.steps//5:l=l+lam*(r-tgt_rate).abs()
        opt.zero_grad();l.backward();opt.step()
    net.eval();ac=0.;fr=0.;nb=20
    with torch.no_grad():
        for _ in range(nb):
            inp,t,m=batch(N,a.S,a.bs,dev);log,f,_=net(inp)
            ac+=(((log.argmax(-1)==t).float()*m).sum()/m.sum()).item();fr+=f
    return ac/nb,fr/nb
def main():
    a=A();dev=torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    grid=[0.03,0.05,0.08,0.12,0.20,0.35];res={"grid":grid,"per_N":[]}
    for N in [4,6,8]:
        ann,_=tr(N,"ann",a,dev)
        curve=[]
        for g in grid:
            ac,fr=tr(N,"snn",a,dev,tgt_rate=g);curve.append({"target":g,"firing":round(fr,4),"acc":round(ac,3)})
            print("N=%d tgt=%.2f -> fire=%.3f acc=%.3f (ANN %.2f)"%(N,g,fr,ac,ann),flush=True)
        res["per_N"].append({"N":N,"ann_acc":round(ann,3),"curve":curve})
    json.dump(res,open(a.out,"w"),indent=2);print("SAVED",a.out,flush=True)
if __name__=="__main__":main()
