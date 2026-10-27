"""GPU implementations of S1 CPU numerical conventions, including ARM f16 FA.

These kernels emulate activation rounding; GGUF weights remain unchanged.
The attention dot reproduces the four 8-lane half accumulators in S1's
ARM NEON ggml_vec_dot_f16; the online value sum also rounds to half at each
update. This differs from ordinary torch SDPA's float32 accumulation.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
CPU_EXP_KEYS=None
CPU_EXP_VALUES=None
CPU_EXP_SIZE=0

@tr.jit
def _cpu_exp(x,KEYS,VALUES,SIZE:tl.constexpr):
    value=libdevice.exp(x.to(tl.float64)).to(tl.float32)
    if SIZE>0:
        if x<0. and x>=-128.:
            bits=x.to(tl.uint32,bitcast=True)&0x7fffffff
            at=(bits*2654435761)&(SIZE-1)
            key=tl.load(KEYS+at)
            while (key!=0)&(key!=bits):
                at=(at+1)&(SIZE-1);key=tl.load(KEYS+at)
            if key==bits:value=tl.load(VALUES+at)
    return value

@tr.jit
def _linear(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr=64,BN:tl.constexpr=64,BK:tl.constexpr=32):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    for b in range(tr.cdiv(K,BK)):
        kk=b*BK+rk
        a=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),0.)
        w=tl.load(W+rn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),0.)
        acc=tl.dot(a,w,acc,input_precision='tf32x3')
    tl.store(Y+rm[:,None]*N+rn[None,:],acc,(rm[:,None]<M)&(rn[None,:]<N))

def linear(x,w):
    x=x.contiguous();shape=x.shape;k=shape[-1];n=w.shape[0];m=x.numel()//k
    y=torch.empty((*shape[:-1],n),device=x.device,dtype=torch.float32)
    _linear[(tr.cdiv(m,64),tr.cdiv(n,64))](x,w,y,m,n,k,num_warps=4)
    return y

@tr.jit
def _q8(X,Y,N:tl.constexpr):
    group=tl.program_id(0);i=tl.arange(0,256)
    x=tl.load(X+group*256+i)
    am=tl.max(tl.abs(x),0)
    first=tl.min(tl.where(tl.abs(x)==am,i,2147483647),0)
    mx=tl.sum(tl.where(i==first,x,0.),0)
    scale=libdevice.div_rn(-127.,mx)
    scale=tl.where(am==0,1.,scale)
    d=libdevice.div_rn(1.,scale)
    q=libdevice.nearbyint(x*scale)
    tl.store(Y+group*256+i,tl.minimum(q,127.)*d)

def q8k(x):
    x=x.contiguous();y=torch.empty_like(x)
    _q8[(x.numel()//256,)](x,y,x.numel(),num_warps=4,enable_fp_fusion=False)
    return y

@tr.jit
def _norm(X,W,Y,D:tl.constexpr,EPS:tl.constexpr,WEIGHT:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(X+r*D+i,i<D,0.)
    mean=(tl.sum((x*x).to(tl.float64),0)/D).to(tl.float32)
    scale=libdevice.div_rn(1.,libdevice.sqrt_rn(mean+EPS))
    y=x*scale
    if WEIGHT:y=y*tl.load(W+i,i<D,0.)
    tl.store(Y+r*D+i,y,i<D)

def rms(x,w=None,eps=1e-6):
    x=x.contiguous();y=torch.empty_like(x);d=x.shape[-1]
    _norm[(x.numel()//d,)](x,x if w is None else w,y,d,eps,w is not None,tr.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return y

@tr.jit
def _gelu(X,T,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);x=tl.load(X+i,i<N,0.)
    bits=x.to(tl.float16).to(tl.uint16,bitcast=True).to(tl.int32)
    y=tl.load(T+bits).to(tl.float32)
    y=tl.where(x<=-10.,0.,tl.where(x>=10.,x,y))
    tl.store(Y+i,y,i<N)

def gelu(x,table):
    x=x.contiguous();y=torch.empty_like(x)
    _gelu[(tr.cdiv(x.numel(),256),)](x,table,y,x.numel(),256,num_warps=4)
    return y

@tr.jit
def _arm_scores(Q,K,S,M,D:tl.constexpr,H:tl.constexpr,T:tl.constexpr,L:tl.constexpr,
                Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,
                K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                M0:tl.constexpr,M2:tl.constexpr,KBL:tl.constexpr):
    r=tl.program_id(0);qpos=r%T;head=(r//T)%H;batch=r//(T*H)
    kk=tl.arange(0,KBL);lane=tl.arange(0,8)
    qbase=Q+batch*Q0+head*Q1+qpos*Q2
    kbase=K+batch*K0+kk[:,None]*K2
    a0=tl.full((KBL,8),0.,tl.float32);a1=a0;a2=a0;a3=a0
    for d in range(D//32):
        ix=d*32+lane
        q0=tl.load(qbase+ix).to(tl.float16).to(tl.float32)
        q1=tl.load(qbase+ix+8).to(tl.float16).to(tl.float32)
        q2=tl.load(qbase+ix+16).to(tl.float16).to(tl.float32)
        q3=tl.load(qbase+ix+24).to(tl.float16).to(tl.float32)
        k0=tl.load(kbase+ix[None,:],kk[:,None]<L,0.)
        k1=tl.load(kbase+ix[None,:]+8,kk[:,None]<L,0.)
        k2=tl.load(kbase+ix[None,:]+16,kk[:,None]<L,0.)
        k3=tl.load(kbase+ix[None,:]+24,kk[:,None]<L,0.)
        a0=tl.fma(k0.to(tl.float64),q0[None,:].to(tl.float64),a0.to(tl.float64)).to(tl.float16).to(tl.float32)
        a1=tl.fma(k1.to(tl.float64),q1[None,:].to(tl.float64),a1.to(tl.float64)).to(tl.float16).to(tl.float32)
        a2=tl.fma(k2.to(tl.float64),q2[None,:].to(tl.float64),a2.to(tl.float64)).to(tl.float16).to(tl.float32)
        a3=tl.fma(k3.to(tl.float64),q3[None,:].to(tl.float64),a3.to(tl.float64)).to(tl.float16).to(tl.float32)
    b0=(a0+a2).to(tl.float16).to(tl.float32)
    b1=(a1+a3).to(tl.float16).to(tl.float32)
    v=(b0+b1).to(tl.float16).to(tl.float32)
    lo=tl.gather(v,tl.broadcast_to(tl.arange(0,4)[None,:],(KBL,4)),1)
    hi=tl.gather(v,tl.broadcast_to((tl.arange(0,4)+4)[None,:],(KBL,4)),1)
    scores=tl.sum(lo+hi,1)
    valid=tl.load(M+batch*M0+qpos*M2+kk,kk<L,False)
    tl.store(S+r*L+kk,tl.where(valid,scores,-float('inf')),kk<L)

@tr.jit
def _arm_values(S,V,O,EXP_KEYS,EXP_VALUES,EXP_SIZE:tl.constexpr,D:tl.constexpr,H:tl.constexpr,T:tl.constexpr,L:tl.constexpr,
                V0:tl.constexpr,V2:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0);batch=r//(T*H);i=tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0.,tl.float32);mx=tl.full((),-float('inf'),tl.float32);den=tl.full((),0.,tl.float32)
    for k in range(L):
        s=tl.load(S+r*L+k)
        if s!=-float('inf'):
            newer=s>mx
            ms=tl.full((),1.,tl.float32);vs=ms
            if newer:ms=_cpu_exp(mx-s,EXP_KEYS,EXP_VALUES,EXP_SIZE)
            else:vs=_cpu_exp(s-mx,EXP_KEYS,EXP_VALUES,EXP_SIZE)
            mx=tl.maximum(mx,s)
            if newer:acc=(acc*ms.to(tl.float16).to(tl.float32)).to(tl.float16).to(tl.float32)
            v=tl.load(V+batch*V0+k*V2+i,i<D,0.)
            acc=tl.fma(v.to(tl.float64),vs.to(tl.float16).to(tl.float64),acc.to(tl.float64)).to(tl.float16).to(tl.float32)
            den=tl.fma(den,ms,vs)
    inv=libdevice.div_rn(1.,den)
    tl.store(O+r*D+i,acc*inv,i<D)

def attention(q,k,v,mask):
    b,h,t,d=q.shape;length=k.shape[2]
    # One KV head in this GGUF; expand() is represented by stride zero.
    if k.shape[0]!=b:k=k.expand(b,-1,-1,-1);v=v.expand(b,-1,-1,-1)
    mask=mask.expand(b,1,t,length)
    scores=torch.empty(b*h*t,length,device=q.device,dtype=torch.float32)
    out=torch.empty(b,h,t,d,device=q.device,dtype=torch.float32)
    _arm_scores[(b*h*t,)](q,k,scores,mask,d,h,t,length,*q.stride()[:3],*k.stride()[:3],
                           mask.stride(0),mask.stride(2),tr.next_power_of_2(length),num_warps=4)
    _arm_values[(b*h*t,)](scores,v,out,CPU_EXP_KEYS if CPU_EXP_SIZE else scores,CPU_EXP_VALUES if CPU_EXP_SIZE else scores,CPU_EXP_SIZE,d,h,t,length,v.stride(0),v.stride(2),tr.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return out
