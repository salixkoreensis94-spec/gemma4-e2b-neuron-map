from config import MODEL
"""S1b pre-softcap lens using equivalent full-vocabulary moments.

E is the dequantized output embedding. For a = RMSNorm(write),
mean(Ea)=mean(E) a; Var(Ea)=a^T Cov(E) a. All vocabulary IDs are included.
Float64 moments avoid the full neurons-by-vocabulary matrix. Direct float32
S1-style logits verify every layer and all final top40 candidates.
"""
import gc,time
import numpy as np
import torch
from gguf import GGUFReader,dequantize
from b_common import *

def run(guard):
 freeze_check();start=time.monotonic();guard()
 reader=GGUFReader(str(MODEL))
 tensors={t.name:t for t in reader.tensors}
 def dq(name):
  t=tensors[name];return torch.from_numpy(dequantize(t.data,t.tensor_type).astype(np.float32,copy=True)).to('cuda')
 E=dq('token_embd.weight');N=dq('output_norm.weight').flatten();V,D=E.shape
 assert (V,D)==(262144,1536)
 groups=load(O/'token_groups.json')['groups'];ids={s:groups[s]['ids'] for s in STYLES}
 # Chunked accumulation limits temporary memory on the configured GPU.
 sm=torch.zeros(D,device='cuda',dtype=torch.float64);gram=torch.zeros(D,D,device='cuda',dtype=torch.float64)
 gt=torch.cuda.Event(enable_timing=True);ge=torch.cuda.Event(enable_timing=True);gt.record()
 for at in range(0,V,8192):
  if at%65536==0:guard()
  part=E[at:at+8192].double();sm+=part.sum(0);gram+=part.T@part
 mu=sm/V;cov=gram/V-mu[:,None]*mu[None,:]
 means=torch.stack([E[ids[s]].double().mean(0) for s in STYLES])
 ge.record();torch.cuda.synchronize();moment_seconds=gt.elapsed_time(ge)/1000
 def scores(v):
  a=v.double()*N.double();av=a@mu
  variance=((a@cov)*a).sum(1).clamp_min(0);sd=variance.sqrt();assert bool((sd>0).all())
  return ((a@means.T)-av[:,None])/sd[:,None]
 def direct(v):
  # S1b RMSNorm in float64 with float32 inverse/products, then float32 GEMM.
  inv=(v.double().square().mean(1)+1e-6).rsqrt().float()
  a=(v*inv[:,None])*N
  z=a@E.T;zz=z.double();m=zz.mean(1);sd=zz.std(1,correction=0)
  return torch.stack([(zz[:,ids[s]].mean(1)-m)/sd for s in STYLES],1)
 allrows=[];timings=[];errors=[]
 for l in range(35):
  guard();freeze_check();p=O/f'lens/layer_{l:02}.npz';meta=p.with_suffix('.json')
  if meta.exists():
   m=load(meta);assert m['rule_sha256']==RULE_HASH and sha(p)==m['sha256']
   data=np.load(p);sc=data['scores'];norms=data['norms'];progress('lens_resume',layer=l)
  else:
   st=time.monotonic();v=dq(f'blk.{l}.ffn_down.weight').T.contiguous()
   sc=scores(v).cpu().numpy();norms=torch.linalg.vector_norm(v,dim=1).cpu().numpy()
   probes=[0,len(v)//2,len(v)-1];err=float((scores(v[probes])-direct(v[probes])).abs().max());errors.append(err)
   assert err<1e-5,('lens direct mismatch',l,err)
   p.parent.mkdir(exist_ok=True)
   with p.open('xb') as f:np.savez_compressed(f,scores=sc,norms=norms)
   secs=time.monotonic()-st
   save(meta,dict(layer=l,neurons=len(v),sha256=sha(p),rule_sha256=RULE_HASH,seconds=secs,direct_max_error=err))
   timings.append(dict(layer=l,neurons=len(v),seconds=secs));del v
   progress('lens_layer_complete',layer=l,neurons=len(sc),seconds=secs,direct_error=err)
   if len(timings)==1:
    estimate=secs*35+moment_seconds
    save(O/'logs/lens_benchmark.json',dict(moment_seconds=moment_seconds,first_layer_seconds=secs,estimated_lens_seconds=estimate,
         reserved_remaining_dev_heldout_seconds=5400,method='Full-vocabulary covariance identity; float64 accumulation, direct float32 checks'))
  for n,scores_row in enumerate(sc):allrows.append(dict(layer=l,neuron=n,scores={s:float(scores_row[i]) for i,s in enumerate(STYLES)}))
 selected={s:sorted(allrows,key=lambda x:(-x['scores'][s],x['layer'],x['neuron']))[:40] for s in STYLES}
 # Recheck every admitted lens candidate against full logits before screening.
 verification=[]
 for l in sorted({x['layer'] for a in selected.values() for x in a}):
  guard();neurons=sorted({x['neuron'] for a in selected.values() for x in a if x['layer']==l})
  v=dq(f'blk.{l}.ffn_down.weight').T[neurons].contiguous()
  for at in range(0,len(v),16):
   err=float((scores(v[at:at+16])-direct(v[at:at+16])).abs().max());assert err<1e-5
   verification.append(dict(layer=l,neurons=neurons[at:at+16],max_error=err))
  del v
 save(O/'lens/top40.json',dict(rule_sha256=RULE_HASH,styles=selected,source='all35 layers/all neuron write columns; descending pre-softcap z-score; layer/neuron ties',
                              direct_verification=verification,seconds=time.monotonic()-start,moment_seconds=moment_seconds))
 del E,N,cov,gram,means,mu,sm,part,reader,tensors;gc.collect();torch.cuda.empty_cache()
 progress('lens_complete',seconds=time.monotonic()-start)
