"""Diagnostic raw-GGUF integer block products matching S1 ARM dot ordering.

Read from the saved S1 build's ggml-cpu/arch/arm/quants.c; no new weights.
"""
import torch,triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
@tr.jit
def _quant(X,Q,SC,K:tl.constexpr):
    b=tl.program_id(0);i=tl.arange(0,256);x=tl.load(X+b*256+i)
    am=tl.max(tl.abs(x),0);first=tl.min(tl.where(tl.abs(x)==am,i,2147483647),0)
    mx=tl.sum(tl.where(i==first,x,0.),0);scale=tl.where(am==0,1.,libdevice.div_rn(-127.,mx))
    d=libdevice.div_rn(1.,scale)
    tl.store(Q+b*256+i,tl.minimum(libdevice.nearbyint(x*scale),127.).to(tl.int8))
    tl.store(SC+b,d)
@tr.jit
def _dot(Q,SC,W,O,MF,K:tl.constexpr,N:tl.constexpr,TY:tl.constexpr,BS:tl.constexpr,FMA:tl.constexpr,R:tl.constexpr=4):
    row=tl.program_id(0)*R+tl.arange(0,R);m=tl.program_id(1);j=tl.arange(0,256)
    acc=tl.full((R,),0.,tl.float32)
    for b in range(K//256):
        base=W+(row*(K//256)+b)*BS
        addr=base[:,None]
        q=tl.load(Q+m*K+b*256+j).to(tl.int32)
        xd=tl.load(SC+m*(K//256)+b)
        if TY==12 or TY==13:
            group=j//32;part=group//2;idx=j%32
            packed=tl.load(addr+4+tl.where(group[None,:]<4,group[None,:],group[None,:]+4),row[:,None]<N,0).to(tl.int32)
            # get_scale_min_k4: low four direct, upper four high two bits.
            low=tl.load(addr+4+(group%4)[None,:],row[:,None]<N,0).to(tl.int32)
            lowmin=tl.load(addr+8+(group%4)[None,:],row[:,None]<N,0).to(tl.int32)
            scale=tl.where(group[None,:]<4,low&63,(packed&15)|((low>>6)<<4))
            mn=tl.where(group[None,:]<4,lowmin&63,(packed>>4)|((lowmin>>6)<<4))
            qs=tl.load(addr+(16 if TY==12 else 48)+(part*32+idx)[None,:],row[:,None]<N,0).to(tl.int32)
            qi=(qs>>((group%2)*4)[None,:])&15
            if TY==13:
                hi=tl.load(addr+16+idx[None,:],row[:,None]<N,0).to(tl.int32)
                qi=qi|(((hi>>group[None,:])&1)<<4)
            d0=tl.load(base.to(tl.pointer_type(tl.float16)),row<N,0).to(tl.float32)
            dm=tl.load((base+2).to(tl.pointer_type(tl.float16)),row<N,0).to(tl.float32)
            isum=tl.sum(q[None,:]*qi*scale,1).to(tl.float32)
            imin=tl.sum(q[None,:]*mn,1).to(tl.float32)
            d=xd*d0;dmin=xd*dm
            if TY==12:
                # S1 CPU_REPACK: four 64-element FMA updates, then bias.
                for sb in tl.static_range(4):
                    si=tl.sum(tl.where(((j//64)==sb)[None,:],q[None,:]*qi*scale,0),1).to(tl.float32)
                    acc=tl.fma(d,si,acc)
                if FMA:acc=tl.fma(-dmin,imin,acc)
                else:acc=acc-dmin*imin
            else:
                if FMA:term=tl.fma(d,isum,-dmin*imin)
                else:term=d*isum-dmin*imin
                acc=acc+term
        else:
            ix=j%128;half=j//128
            low=tl.load(addr+(half*64+ix%64)[None,:],row[:,None]<N,0).to(tl.int32)
            hi=tl.load(addr+128+(half*32+ix%32)[None,:],row[:,None]<N,0).to(tl.int32)
            qi=((low>>((ix//64)*4)[None,:])&15)|(((hi>>((ix//32)*2)[None,:])&3)<<4)
            scale=tl.load((addr+192+(j//16)[None,:]).to(tl.pointer_type(tl.int8)),row[:,None]<N,0).to(tl.int32)
            isum=tl.sum(q[None,:]*(qi-32)*scale,1).to(tl.float32)
            wd=tl.load((base+208).to(tl.pointer_type(tl.float16)),row<N,0).to(tl.float32)
            d=wd*xd
            if FMA:
                fused=tl.load(MF+m)
                acc=tl.where(fused,tl.fma(d,isum,acc),acc+d*isum)
            else:acc=acc+d*isum
    tl.store(O+m*N+row,acc,row<N)
@tr.jit
def _bf(X,W,Y,K:tl.constexpr,N:tl.constexpr,B:tl.constexpr,R:tl.constexpr=4):
    m=tl.program_id(1);r=tl.program_id(0)*R+tl.arange(0,R);j=tl.arange(0,B)
    x=tl.load(X+m*K+j,j<K,0).to(tl.bfloat16).to(tl.float64)
    w=tl.load(W+r[:,None]*K+j[None,:],(r[:,None]<N)&(j[None,:]<K),0).to(tl.float64)
    y=tl.sum(x[None,:]*w,1).to(tl.float32)
    tl.store(Y+m*N+r,y,r<N)
@tr.jit
def _f32(X,W,Y,SG,K:tl.constexpr,N:tl.constexpr,R:tl.constexpr=4):
    m=tl.program_id(1);r=tl.program_id(0)*R+tl.arange(0,R);lane=tl.arange(0,4)
    a0=tl.full((R,4),0.,tl.float32);a1=a0;a2=a0;a3=a0
    if tl.load(SG+m):
        for b in range(K//4):
            j=b*4+lane
            x=tl.load(X+m*K+j);w=tl.load(W+r[:,None]*K+j[None,:],r[:,None]<N,0)
            a0=tl.fma(x[None,:],w,a0)
    else:
        for b in range(K//16):
            j=b*16+lane
            x=tl.load(X+m*K+j);w=tl.load(W+r[:,None]*K+j[None,:],r[:,None]<N,0);a0=tl.fma(x[None,:],w,a0)
            x=tl.load(X+m*K+j+4);w=tl.load(W+r[:,None]*K+j[None,:]+4,r[:,None]<N,0);a1=tl.fma(x[None,:],w,a1)
            x=tl.load(X+m*K+j+8);w=tl.load(W+r[:,None]*K+j[None,:]+8,r[:,None]<N,0);a2=tl.fma(x[None,:],w,a2)
            x=tl.load(X+m*K+j+12);w=tl.load(W+r[:,None]*K+j[None,:]+12,r[:,None]<N,0);a3=tl.fma(x[None,:],w,a3)
        a0=(a0+a2)+(a1+a3)
    even=tl.gather(a0,tl.broadcast_to((tl.arange(0,2)*2)[None,:],(R,2)),1)
    odd=tl.gather(a0,tl.broadcast_to((tl.arange(0,2)*2+1)[None,:],(R,2)),1)
    y=tl.sum(even+odd,1)
    tl.store(Y+m*N+r,y,r<N)
def linear(x,t,raw,w,fma=False,row_fma=None,sgem=None):
    x=x.contiguous();k=x.shape[-1];n=w.shape[0];m=x.numel()//k
    y=torch.empty((*x.shape[:-1],n),device='cuda',dtype=torch.float32);ty=int(t.tensor_type)
    if ty==0:
        if sgem is None:sgem=torch.full((m,),x.ndim>=3 and x.shape[-2]>=4,device='cuda',dtype=torch.bool)
        _f32[(tr.cdiv(n,4),m)](x,w,y,sgem.contiguous().reshape(-1),k,n,enable_fp_fusion=False)
    elif ty==30:
        _bf[(tr.cdiv(n,4),m)](x,w,y,k,n,tr.next_power_of_2(k),enable_fp_fusion=False)
    else:
        q=torch.empty_like(x,dtype=torch.int8);sc=torch.empty(x.numel()//256,device='cuda')
        _quant[(x.numel()//256,)](x,q,sc,k,enable_fp_fusion=False)
        bs={12:144,13:176,14:210}[ty]
        if row_fma is None:
            length=x.shape[-2] if x.ndim>=3 else 1
            row_fma=(torch.arange(m,device='cuda')%length)<(length//4*4)
        row_fma=row_fma.contiguous().reshape(-1)
        _dot[(tr.cdiv(n,4),m)](q,sc,raw,y,row_fma,k,n,ty,bs,fma,enable_fp_fusion=False,num_warps=4)
    return y

@tr.jit
def _rope(X,C,Y,N:tl.constexpr,D:tl.constexpr,H:tl.constexpr,T:tl.constexpr,POS,B:tl.constexpr,MODE:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    col=i%D;headrow=i//D;token=headrow//H
    p=tl.load(POS+token,token<(N//(D*H)),0).to(tl.int32)
    cs=tl.load(C+p*(D*2)+col*2);sn=tl.load(C+p*(D*2)+col*2+1)
    a=tl.load(X+headrow*(D*2)+col,valid,0);b=tl.load(X+headrow*(D*2)+D+col,valid,0)
    if MODE&1:y0=tl.fma(-b,sn,a*cs)
    else:y0=tl.fma(a,cs,-b*sn)
    if MODE&2:y1=tl.fma(b,cs,a*sn)
    else:y1=tl.fma(a,sn,b*cs)
    tl.store(Y+headrow*(D*2)+col,y0,valid);tl.store(Y+headrow*(D*2)+D+col,y1,valid)
def rope(x,pos,cache,mode=0):
    x=x.contiguous();pos=pos.contiguous();y=torch.empty_like(x);d=x.shape[-1]//2;n=x.numel()//2
    # cache stride is 2*d (cos/sin interleaved).
    _rope[(tr.cdiv(n,256),)](x,cache,y,n,d,x.shape[-2],x.shape[-3],pos,256,mode,enable_fp_fusion=False)
    return y
