"""Phase-2: is the sparsity ceiling RNN-specific? Spiking Transformer vs matched ANN Transformer.
Attention has random-access to history (KV), not a compressed recurrent spike state, so the Prop-1
compression argument need not apply. Test: does spiking-attention firing floor look like the RNN
(~50%, cannot sparsify) or like feed-forward (low, sparsifiable)?
Char-level WikiText-103 (local). Spiking neurons = FFN-hidden + attention-output LIF, integrated over
T_s spike-steps. Two-sided probe on mean firing."""
import json,argparse,math,torch,torch.nn as nn,torch.nn.functional as F
import snntorch as snn; from snntorch import surrogate
ARROW="/work/zeyuwang/5duaa/data_cache/wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow"
def A():
    p=argparse.ArgumentParser();p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--E",type=int,default=128);p.add_argument("--heads",type=int,default=4)
    p.add_argument("--L",type=int,default=64);p.add_argument("--Ts",type=int,default=4)
    p.add_argument("--bs",type=int,default=64);p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--chars",type=int,default=400000);p.add_argument("--lr",type=float,default=3e-3)
    p.add_argument("--mode",default="snn");p.add_argument("--lam",type=float,default=0.0)
    p.add_argument("--target",type=float,default=0.1)
    p.add_argument("--out",default="/work/zeyuwang/neuro_poc/paper_runs/spk_tr.json");return p.parse_args()
def load(nchars,L,dev):
    from datasets import Dataset
    txt="\n".join(t for t in Dataset.from_file(ARROW)["text"] if t and t.strip())[:nchars]
    ch=sorted(set(txt));V=len(ch);stoi={c:i for i,c in enumerate(ch)}
    ids=torch.tensor([stoi[c] for c in txt],dtype=torch.long)
    n=(len(ids)-1)//L;x=ids[:n*L].view(n,L);y=ids[1:n*L+1].view(n,L);cut=int(n*.9)
    return (x[:cut],y[:cut]),(x[cut:],y[cut:]),V
class Block(nn.Module):
    def __init__(s,E,H,mode):
        super().__init__();s.E=E;s.H=H;s.mode=mode
        s.ln1=nn.LayerNorm(E);s.qkv=nn.Linear(E,3*E);s.proj=nn.Linear(E,E)
        s.ln2=nn.LayerNorm(E);s.ff1=nn.Linear(E,4*E);s.ff2=nn.Linear(4*E,E)
        if mode=="snn":
            s.la=snn.Leaky(beta=.9,spike_grad=surrogate.fast_sigmoid(slope=25))
            s.lf=snn.Leaky(beta=.9,spike_grad=surrogate.fast_sigmoid(slope=25))
    def attn(s,x):
        B,L,E=x.shape;qkv=s.qkv(x).reshape(B,L,3,s.H,E//s.H).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.proj(o.transpose(1,2).reshape(B,L,E))
class SpkTransformer(nn.Module):
    def __init__(s,V,E,H,L,Ts,mode):
        super().__init__();s.mode=mode;s.Ts=Ts if mode=="snn" else 1
        s.emb=nn.Embedding(V,E);s.pos=nn.Parameter(torch.zeros(1,L,E));s.blk=Block(E,H,mode)
        s.lnf=nn.LayerNorm(E);s.head=nn.Linear(E,V)
    def forward(s,x):
        h0=s.emb(x)+s.pos[:,:x.size(1)]
        if s.mode=="ann":
            a=s.blk.attn(s.blk.ln1(h0));h=h0+a
            f=s.blk.ff2(F.gelu(s.blk.ff1(s.blk.ln2(h))));h=h+f
            return s.head(s.lnf(h)),0.0,None
        ma=s.blk.la.init_leaky();mf=s.blk.lf.init_leaky();out=0.;fire=0.;racc=0.;cnt=0
        for _ in range(s.Ts):
            a=s.blk.attn(s.blk.ln1(h0));sa,ma=s.blk.la(a,ma)   # attention-output spikes
            fire+=sa.mean().item();racc=racc+sa.mean();cnt+=1
            h=h0+sa
            ff=s.blk.ff1(s.blk.ln2(h));sf,mf=s.blk.lf(ff,mf)   # FFN-hidden spikes
            fire+=sf.mean().item();racc=racc+sf.mean();cnt+=1
            h=h+s.blk.ff2(sf)
            out=out+s.head(s.lnf(h))
        return out/s.Ts,fire/cnt,racc/cnt
def run(mode,a,dev,tr,va,V,lam=0.,tgt=0.1):
    net=SpkTransformer(V,a.E,a.heads,a.L,a.Ts,mode).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr);ce=nn.CrossEntropyLoss()
    xt,yt=tr;nb=xt.size(0)//a.bs
    for ep in range(a.epochs):
        net.train();perm=torch.randperm(xt.size(0));last=0.
        for i in range(nb):
            idx=perm[i*a.bs:(i+1)*a.bs];x=xt[idx].to(dev);y=yt[idx].to(dev)
            log,_,r=net(x);l=ce(log.reshape(-1,V),y.reshape(-1))
            if mode=="snn" and lam>0 and ep>=1:l=l+lam*(r-tgt).abs()
            opt.zero_grad();l.backward();opt.step();last=l.item()
        print("[%s lam=%.2f] ep%d loss %.3f"%(mode,lam,ep+1,last),flush=True)
    net.eval();tot=0.;nt=0;fr=0.;nbv=0;xv,yv=va
    with torch.no_grad():
        for i in range(xv.size(0)//a.bs):
            x=xv[i*a.bs:(i+1)*a.bs].to(dev);y=yv[i*a.bs:(i+1)*a.bs].to(dev)
            log,f,_=net(x);tot+=ce(log.reshape(-1,V),y.reshape(-1)).item()*y.numel();nt+=y.numel();fr+=f;nbv+=1
    return math.exp(tot/nt),(tot/nt)/math.log(2),fr/max(nbv,1)
def main():
    a=A();dev=torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    tr,va,V=load(a.chars,a.L,dev);print("V=%d Ts=%d"%(V,a.Ts),flush=True)
    ann=run("ann",a,dev,tr,va,V)
    res={"ann_bpc":round(ann[1],4),"ann_ppl":round(ann[0],2),"snn":[]}
    for lam,tgt in [(0.,0.1),(1.0,0.05),(3.0,0.02)]:
        ppl,bpc,fire=run("snn",a,dev,tr,va,V,lam=lam,tgt=tgt)
        row={"lam":lam,"target":tgt,"bpc":round(bpc,4),"ppl":round(ppl,2),"firing":round(fire,4)}
        res["snn"].append(row);print("SNN lam=%.1f tgt=%.2f -> bpc %.3f fire %.3f"%(lam,tgt,bpc,fire),flush=True)
    json.dump(res,open(a.out,"w"),indent=2);print("SAVED",json.dumps(res),flush=True)
if __name__=="__main__":main()
